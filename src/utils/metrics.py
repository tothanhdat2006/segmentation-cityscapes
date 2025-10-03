import numpy as np

import torch

class CustomMeanIoU:
    def __init__(self, num_classes: int, ignore_index: int = None, device: torch.device = None):
        """
        Custom MeanIoU metric that explicitly handles an ignore_index.

        Args:
            num_classes (int): The number of valid classes (excluding the ignore_index).
            ignore_index (int, optional): The class label to ignore in calculations. Defaults to None.
            device (torch.device, optional): The device to store internal tensors on. Defaults to None (CPU).
        """
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device if device is not None else torch.device("cpu")

        self.intersections = torch.zeros(num_classes, dtype=torch.long, device=self.device)
        self.unions = torch.zeros(num_classes, dtype=torch.long, device=self.device)
        
        self._num_updates = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric state with new predictions (logits) and targets (one-hot masks).

        Args:
            preds (torch.Tensor): Predicted mask logits (float tensor, shape [B, C, H, W]).
            target (torch.Tensor): Ground truth semantic masks (long/float tensor, shape [B, C, H, W] - assumed one-hot or similar).
                                   Values should represent class membership.
                                   If target is already class IDs [B, H, W], please adjust this function.
        """
        # --- 1. Convert logits to predicted class IDs [B, H, W] ---
        if preds.dim() != 4:
            raise ValueError(f"Preds must be [B, C, H, W] where C is num_classes. Got {preds.shape}")
        predicted_classes = torch.argmax(preds, dim=1).to(torch.long)

        # --- 2. Convert target from multi-channel (e.g., one-hot) to class IDs [B, H, W] ---
        if target.dim() == 4:
            if target.shape[1] != 1:
                target_classes = torch.argmax(target, dim=1).clone().to(torch.long)
            else:
                target_classes = target.squeeze(1).clone().to(torch.long) # Added .clone() for safety
        elif target.dim() == 3:
            target_classes = target.clone().to(torch.long)
        else:
            raise ValueError(
                f"Unsupported target shape or format. Expected [B, C, H, W] (one-hot), "
                f"[B, 1, H, W], or [B, H, W] (class IDs). Got {target.shape}"
            )

        if predicted_classes.shape != target_classes.shape:
            raise ValueError(
                f"Predicted and target class maps must have the same spatial shape. "
                f"Got predicted: {predicted_classes.shape} and target: {target_classes.shape}"
            )
        
        predicted_classes = predicted_classes.to(self.device)
        target_classes = target_classes.to(self.device)

        valid_mask = torch.ones_like(target_classes, dtype=torch.bool, device=self.device)
        if self.ignore_index is not None:
            valid_mask = (target_classes != self.ignore_index)

        preds_valid = predicted_classes[valid_mask]
        target_valid = target_classes[valid_mask]
        
        for c in range(self.num_classes):
            target_c = (target_valid == c)
            preds_c = (preds_valid == c)

            intersection = (preds_c & target_c).sum()
            union = (preds_c | target_c).sum()

            self.intersections[c] += intersection
            self.unions[c] += union
            
        self._num_updates += preds.shape[0]

    def compute(self) -> float:
        """
        Compute the Mean IoU based on accumulated intersections and unions.

        Returns:
            float: The Mean IoU value. Returns 0.0 if no valid pixels were processed.
        """
        ious = torch.zeros(self.num_classes, dtype=torch.float, device=self.device)

        for c in range(self.num_classes):
            if self.unions[c] > 0:
                ious[c] = self.intersections[c].float() / self.unions[c].float()
            else:
                # If a class had no ground truth or predictions in valid regions, its IoU is NaN
                # We will ignore these NaNs when computing the mean.
                ious[c] = torch.nan 

        # Filter out NaN values (classes with no occurrences) for mean calculation
        valid_ious = ious[~torch.isnan(ious)]
        # print(ious)
        if len(valid_ious) == 0:
            return 0.0
            
        mean_iou = torch.mean(valid_ious)
        return mean_iou.item()

    def reset(self):
        """
        Reset the metric state.
        """
        self.intersections = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.unions = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self._num_updates = 0



# Original: https://github.com/matterport/Mask_RCNN
def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
        
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)
    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    
    return overlaps

# Original: https://github.com/matterport/Mask_RCNN
def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]
    
# Original: https://github.com/matterport/Mask_RCNN
def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.1): # low score threshold
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    # gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    # pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1].copy()
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)
    # print(overlaps.shape)
    # plt.figure(figsize=(15, 18))
    # plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.colorbar()
    
    # tick_marks_gt = np.arange(len(gt_class_ids))
    # tick_marks = np.arange(len(pred_class_ids))
    # plt.xticks(tick_marks_gt, gt_class_ids, rotation=45)
    # plt.yticks(tick_marks, pred_class_ids)
    
    # # Adding text annotations
    # for i in range(overlaps.shape[0]):
    #     for j in range(overlaps.shape[1]):
    #         plt.text(j, i, format(overlaps[i, j], 'f'),
    #                  horizontalalignment="center",
    #                  color="white" if overlaps[i, j] > overlaps.max() / 2 else "black")
    
    # plt.ylabel('Predicted Label')
    # plt.xlabel('True Label')
    # plt.tight_layout()
    # plt.show()

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


# Original: https://github.com/matterport/Mask_RCNN
def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    gt_boxes, pred_boxes : (N, 4)
    gt_class_ids, pred_class_ids : (N)
    gt_masks, pred_masks : (H, W, N)

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps

# Original: https://github.com/matterport/Mask_RCNN
def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)
    
    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP