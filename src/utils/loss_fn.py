import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self,  epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        
    @staticmethod
    def _calculate_dice_coeff(input_mask: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        """
        Calculates the Dice coefficient for binary masks.
        Args:
            input_mask (Tensor): Predicted mask, expected to be probabilities (float).
            target (Tensor): Ground truth mask, expected to be binary (0 or 1, float).
            reduce_batch_first (bool): If True, computes a single Dice score over the entire batch
                                       (B, H, W) -> scalar. If False, computes Dice per item (B, H, W) -> B scores.
            epsilon (float): Small constant to avoid division by zero.
        Returns:
            Tensor: Dice coefficient.
        """
        # print(input_mask.size(), target.size())
        assert input_mask.size() == target.size(), f"Input and target must have the same size. Got {input_mask.size()} and {target.size()}"
        assert input_mask.dim() >= 2, "Input mask must have at least 2 dimensions (H, W)"
        assert input_mask.dim() == 3 or not reduce_batch_first, \
            "If reduce_batch_first is True, input_mask must be 3-dimensional (B, H, W) after potential flattening."
    
        # Determine dimensions to sum over
        if reduce_batch_first: # Sum over B, H, W
            sum_dim = (-1, -2, -3) # Assumes B, H, W for input_mask.dim() == 3, or C, H, W after flatten(0,1) from B, C, H, W
        else: # Sum over H, W (per item in batch/channel)
            sum_dim = (-1, -2)
    
        inter = 2 * (input_mask * target).sum(dim=sum_dim)  # 2 * |A giao B|
        sets_sum = input_mask.sum(dim=sum_dim) + target.sum(dim=sum_dim)  # |A| + |B|
        
        # Handle cases where sets_sum is 0 to avoid NaNs, set dice to 1 if both are empty (inter=0, sets_sum=0)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
        #print("sets_sum = ", sets_sum)
        dice = (inter + epsilon) / (sets_sum + epsilon)  # (2 * |A giao B| + eps) / (|A| + |B| + eps)
        return dice.mean()
    
    def _multiclass_dice_coeff(self, input_mask: Tensor, target: Tensor, reduce_batch_first: bool = False):
        """
        Calculates the average Dice coefficient across all classes for multi-class segmentation.
        Flattens (B, C, H, W) to (B*C, H, W) and calls dice_coeff.
        Args:
            input_mask (Tensor): Predicted masks, shape (B, C, H, W), probabilities (float).
            target (Tensor): Ground truth masks, shape (B, C, H, W), one-hot encoded (float).
            reduce_batch_first (bool): If True, computes a single Dice score over the entire (B*C) flattened batch.
                                       If False, computes Dice per (B*C) item.
            epsilon (float): Small constant to avoid division by zero.
        Returns:
            Tensor: Average Dice coefficient.
        """
        # Flatten N and C dimensions to treat each class in each batch as a separate binary mask
        # Shape changes from (N, C, H, W) to (N*C, H, W)
        return self._calculate_dice_coeff(
            input_mask.flatten(0, 1), 
            target.flatten(0, 1), 
            reduce_batch_first, 
            self.epsilon
        )

    def _binary_dice_coeff(self, input_mask: Tensor, target: Tensor, reduce_batch_first: bool) -> Tensor:
        return self._calculate_dice_coeff(
            input_mask, 
            target, 
            reduce_batch_first, 
            self.epsilon
        )
    
    def forward(self, input_mask: Tensor, target: Tensor, multiclass: bool = False):
        """
        Calculates the Dice loss (objective to minimize) between 0 and 1.
        Args:
            input_mask (Tensor): Predicted mask (logits or probabilities) (B, C, H, W).
            target (Tensor): Ground truth mask (binary or one-hot encoded)  (B, C, H, W).
            multiclass (bool): If True, uses multiclass_dice_coeff; otherwise, uses dice_coeff.
        Returns:
            Tensor: Dice loss (1 - Dice_coefficient).
        """
        # Note: If input_mask are logits, they should be passed through softmax/sigmoid before
        # being used with dice_coeff/multiclass_dice_coeff for optimal results,
        # as these functions expect probabilities or binary masks.
        fn = self._multiclass_dice_coeff if multiclass else self._binary_dice_coeff
        
        if multiclass:
            input_probs = F.softmax(input_mask, dim=1)
            if input_probs.shape[1] > target.shape[1]:
                input_probs = input_probs[:, 1:, ...]
        elif input_mask.dim() == target.dim(): # Binary case, e.g., (N, H, W)
            input_probs = torch.sigmoid(input_mask)
        else:
            input_probs = input_mask
    
        return 1 - fn(input_probs, target, reduce_batch_first=True)

class FocalLoss(nn.Module):
    def __init__(self, ignore_index=0, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Logits from model (B, C, H, W).
        targets: Ground truth masks (B, H, W).
        """
        if targets.size() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1).long()

        CE_loss = F.cross_entropy(inputs, targets, reduction=self.reduction, ignore_index=self.ignore_index)
        # focal loss = BCE * a * (1 - e^(-BCE))^y 
        pt = torch.exp(-CE_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Combined Dice + Focal + CE Loss
class DiceFocalCELoss(nn.Module):
    def __init__(self, weights = [0.35, 0.3, 0.35], ignore_index=0, alpha=0.35, gamma=3, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(ignore_index=ignore_index, alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(self, inputs, onehot_targets, semantic_targets):
        """
        inputs: Logits from model (B, C, H, W).
        onehot_targets: Ground truth masks (B, C-1, H, W).
        semantic_targets: Ground truth masks (B, H, W).
        """
        inputs = inputs.float()
        onehot_targets = onehot_targets.long()
        semantic_targets = semantic_targets.long()

        losses = {
            'ce': 0.0,
            'dl': 0.0,
            'fc': 0.0
        }
        losses['ce'] = self.weights[0] * self.ce_loss(inputs, semantic_targets)  
        losses['dl'] = self.weights[1] * self.dice_loss(inputs, onehot_targets, multiclass=True)
        losses['fc'] = self.weights[2] * self.focal_loss(inputs, semantic_targets)
        
        return losses
    

