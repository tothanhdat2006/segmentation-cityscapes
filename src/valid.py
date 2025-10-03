from tqdm import tqdm
import numpy as np
import argparse

import torch
import torchvision.transforms.v2 as T
from torchmetrics.segmentation import MeanIoU

from models.model import load_maskrcnn, load_unet
from utils.metrics import CustomMeanIoU, compute_ap_range
from utils.visualize import convert_instance_to_semantic
from datasets.dataset import CityscapesDataset, get_dataloader
from configs.config import config, id_to_trainId_map_20c, id_to_trainId_map_19c, id_to_trainId_map_9c, id_to_trainId_map_8c

def evaluate_maskrcnn_model(model, dataloader, device, num_classes, mIoU_metric):
    print('Evaluating Mask R-CNN for Semantic mIoU...')
    model.eval()
    
    mAP = 0.0
    mAP50 = 0.0
    cnt = 0
    no_lbl_9c = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), leave=False) as pbar:
            for images, targets in dataloader:
                images = list(image.to(device) for image in images)
                img_height, img_width = images[0].shape[-2:]
                cnt += len(images)
                
                predictions = model(images)

                # ===================================================== Calculate mIoU ====================================================
                # Convert prediction masks and target masks to corresponding semantic masks
                # in order to calculate mIoU
                pred_masks_batch = []
                gt_masks_batch = []
                for i in range(len(images)):
                    if targets[i]['labels'].shape[0] == 1: # exist image without person or vehicle
                        print(targets[i]['labels'])
                        continue
                    # img_height, img_width = images[i].shape[-2:] 
                    
                    predicted_class_map = torch.full(
                        (img_height, img_width),
                        fill_value=0, 
                        dtype=torch.long,
                        device=device
                    )
                    image_predictions = predictions[i]
                    pred_masks_i = image_predictions['masks'].clone()   # (N_pred, 1, H, W)
                    pred_labels_i = image_predictions['labels'].clone() # (N_pred)
                    pred_scores_i = image_predictions['scores'].clone() # (N_pred)
                    if pred_masks_i.numel() > 0:
                        # (N_pred, 1, H, W) -> (N_pred, H, W)
                        pred_masks_i = pred_masks_i.squeeze(1)

                        # Sort predictions by score in ascending order
                        sorted_indices = torch.argsort(pred_scores_i, descending=False)
                        pred_masks_i = pred_masks_i[sorted_indices]
                        pred_labels_i = pred_labels_i[sorted_indices]

                        for j in range(pred_masks_i.shape[0]):
                            mask_j = (pred_masks_i[j] >= 0.5)
                            label_j = pred_labels_i[j]

                            if label_j < num_classes:
                                predicted_class_map[mask_j] = label_j # gradually replaced by class with higher score
                            
                    pred_masks_batch.append(predicted_class_map)

                    ground_truth_semantic_map = torch.full(
                        (img_height, img_width),
                        fill_value=0,
                        dtype=torch.long,
                        device=device
                    )
                    image_targets = targets[i]
                    gt_masks_i = image_targets['masks'].clone()   # (N_gt, H, W) (0 or 1)
                    gt_labels_i = image_targets['labels'].clone() # (N_gt)
                    if gt_masks_i.numel() > 0:
                        for j in range(gt_masks_i.shape[0]):
                            mask_j = (gt_masks_i[j] == 1)
                            label_j = gt_labels_i[j]

                            if label_j < num_classes:
                                ground_truth_semantic_map[mask_j] = label_j

                    gt_masks_batch.append(ground_truth_semantic_map)

                pred_masks_batch = torch.stack(pred_masks_batch).cpu()
                gt_masks_batch = torch.stack(gt_masks_batch).cpu()
                
                if len(pred_masks_batch) > 0 and len(gt_masks_batch) > 0:
                    mIoU_metric.update(pred_masks_batch, gt_masks_batch)
                    
                # ===================================================== Calculate mAP =====================================================
                for i in range(len(predictions)):
                    if targets[i]['labels'].shape[0] == 1: # exist image without person or vehicle
                        no_lbl_9c += 1
                        continue
                    target_mask = targets[i]['masks'][1:, ...].to('cpu').permute(1, 2, 0) # [1:] to remove unlabel class (which is 255 in trainId)
                    target_mask = np.array(target_mask).astype(bool)
                    targets[i] = {k: np.array(v[1:, ...].to('cpu')) for k, v in targets[i].items()}
                    
                    pred = predictions[i]
                    pred_scores = pred['scores'].cpu().numpy()
                    high_conf_indices = np.where(pred_scores >= 0.5)[0]
                    pred_scores = pred_scores[high_conf_indices]
                    pred_masks = (pred['masks'][high_conf_indices] > 0.5).squeeze(1).permute(1, 2, 0).cpu().numpy()
                    pred_labels = pred['labels'][high_conf_indices].cpu().numpy()
                    pred_boxes = pred['boxes'][high_conf_indices].cpu().numpy()

                    # pred_labels = (lambda x: x-1)(pred_labels) # from 1-indexed to 0-indexed
                    # targets[i]['labels'] = (lambda x: x-1)(targets[i]['labels']) # from 1-indexed to 0-indexed

                    mAP50 += compute_ap_range(targets[i]['boxes'], targets[i]['labels'], target_mask,
                                        pred_boxes, pred_labels, pred_scores, pred_masks,
                                        iou_thresholds=[0.5], verbose=0)
                    
                    mAP += compute_ap_range(targets[i]['boxes'], targets[i]['labels'], target_mask,
                                        pred_boxes, pred_labels, pred_scores, pred_masks,
                                        iou_thresholds=None, verbose=0)
                    
                pbar.update(1)
                pbar.set_postfix(iou=f"{mIoU_metric.compute():.4f}", mAP50=f"{mAP50/(cnt - no_lbl_9c):.2f}", mAP=f"{mAP/(cnt - no_lbl_9c):.2f}")

    val_miou = mIoU_metric.compute()
    mAP50 /= (cnt - no_lbl_9c)
    mAP /= (cnt - no_lbl_9c)
    model.train()
    return val_miou, mAP50, mAP

def evaluate_unet_model(model, dataloader, device, num_classes, mIoU_metric):
    """
    Correctly evaluates a U-Net model using MeanIoU, ignoring the class index 255.
    """
    print('Evaluating U-Net model with MeanIoU...')
    model.eval()

    cnt = 0
    mAP = 0.0
    mAP50 = 0.0
    no_lbl_8c = 0
    with torch.no_grad():
        with tqdm(desc="Evaluating U-Net", total=len(dataloader), leave=False) as pbar:
            for images, targets in dataloader:
                images = torch.stack(images).to(device) # [B, 3, H, W]
                cnt += len(images)     
                semantic_masks = [convert_instance_to_semantic(t['masks'], t['labels']) for t in targets]
                semantic_masks = torch.stack(semantic_masks).squeeze(1) # [B, H, W]

                predicted_masks = model(images) # [B, 19, H, W]
                # ===================================================== Calculate mIoU ====================================================
                onehot_masks = torch.zeros((semantic_masks.shape[0], num_classes, semantic_masks.shape[1], semantic_masks.shape[2]), dtype=torch.long)
                for i in range(semantic_masks.shape[0]):
                    if targets[i]['labels'].shape[0] == 1: # exist image without person or vehicle
                        continue
                    unique_classes_in_this_mask = torch.unique(semantic_masks[i])
                    for class_id in range(num_classes):
                        if class_id in unique_classes_in_this_mask:
                            onehot_masks[i, class_id, :, :] = (semantic_masks[i] == class_id).long()

                mIoU_metric.update(predicted_masks.long().cpu(), onehot_masks.long().cpu())

                # ===================================================== Calculate mAP =====================================================
                pred_boxes = np.zeros((num_classes, 4)) # dummy mAP doesnt process what inside
                pred_scores = np.full(num_classes, 1.0) # assume model is sure about the mask
                pred_labels = np.arange(num_classes)
                for i in range(len(predicted_masks)):
                    if targets[i]['labels'].shape[0] == 1: # exist image without person or vehicle
                        print(targets[i]['labels'])
                        no_lbl_8c += 1
                        continue
                    target_mask = targets[i]['masks'][:-1, ...].to('cpu').permute(1, 2, 0)
                    target_mask = np.array(target_mask).astype(bool)
                    
                    targets[i] = {k: np.array(v[:-1, ...].to('cpu')) for k, v in targets[i].items()}

                    pred_masks = (predicted_masks[i] > 0.5).to('cpu').permute(1, 2, 0)
                    pred_masks = np.array(pred_masks)
                    mAP50 += compute_ap_range(targets[i]['boxes'], targets[i]['labels'], target_mask,
                                        pred_boxes, pred_labels, pred_scores, pred_masks,
                                        iou_thresholds=[0.5], verbose=0)
                    
                    mAP += compute_ap_range(targets[i]['boxes'], targets[i]['labels'], target_mask,
                                        pred_boxes, pred_labels, pred_scores, pred_masks,
                                        iou_thresholds=None, verbose=0)
                    
                pbar.update(1)
                pbar.set_postfix(iou=f"{mIoU_metric.compute():.4f}", mAP50=f"{mAP50/(cnt - no_lbl_8c):.4f}", mAP=f"{mAP/(cnt - no_lbl_8c):.4f}")
                # break

    val_miou = mIoU_metric.compute()
    mAP50 /= (cnt - no_lbl_8c)
    mAP /= (cnt - no_lbl_8c)
    model.train()
    return val_miou, mAP50, mAP

def validate_maskrcnn(valid_type, model_ckpt_path, config):
    val_augmentation_maskrcnn = T.Compose([
        T.Resize((512,1024)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if valid_type == "full":
        mIoU_metric = MeanIoU(num_classes=20, include_background=False)
        id_to_trainId_map = id_to_trainId_map_20c
        num_classes = 20
    else:
        mIoU_metric = MeanIoU(num_classes=9, include_background=False)
        id_to_trainId_map = id_to_trainId_map_9c
        num_classes = 9
    
    model = load_maskrcnn(num_classes, model_ckpt_path).to(config.device)
    val_dataset = CityscapesDataset(config, id_to_trainId_map, ignore_index=0, transform=val_augmentation_maskrcnn, split='val')
    val_dataloader = get_dataloader(val_dataset, config, is_train=False)
    val_miou, val_mAP50, val_mAP = evaluate_maskrcnn_model(model, val_dataloader, config.device, num_classes, mIoU_metric)
    return val_miou, val_mAP50, val_mAP
        
def validate_unet(valid_type, model_ckpt_path, config):
    val_augmentation_unet = T.Compose([
        T.Resize((512,1024)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if valid_type == "full":
        mIoU_metric = CustomMeanIoU(
            num_classes=19,
            ignore_index=255
        )
        id_to_trainId_map = id_to_trainId_map_19c
        num_classes = 19
    else:
        mIoU_metric = CustomMeanIoU(
            num_classes=8,
            ignore_index=255
        )
        id_to_trainId_map = id_to_trainId_map_8c
        num_classes = 8
    
    model = load_unet(num_classes, model_ckpt_path).to(config.device)
    val_dataset = CityscapesDataset(config, id_to_trainId_map, ignore_index=255, transform=val_augmentation_unet, split='val')
    val_dataloader = get_dataloader(val_dataset, config, is_train=False)
    val_miou, val_mAP50, val_mAP = evaluate_unet_model(model, val_dataloader, config.device, num_classes, mIoU_metric)
    return val_miou, val_mAP50, val_mAP

if __name__ == "__main__":    
    parser = argparse.ArgumentParser("Validating the model performance")
    parser.add_argument("-m", "--model", choices=["maskrcnn", "unet"], type=str, help="Model name (MaskRCNN or UNet)")
    parser.add_argument("-t", "--type", choices=["full", "pedveh"], type=str, help="Full or person+vehicle")
    parser.add_argument("--ckpt_path", type=str, help="Model checkpoint path")
    args = parser.parse_args()
    if args.model == "maskrcnn":
        val_miou, val_mAP50, val_mAP = validate_maskrcnn(args.type, args.ckpt_path, config)
        print("Validation mIoU: ", val_miou)
        print("Validation mAP@50: ", val_mAP50)
        print("Validation mAP: ", val_mAP)
    elif args.model == "unet":
        val_miou, val_mAP50, val_mAP = validate_unet(args.type, args.ckpt_path, config)
        print("Validation mIoU: ", val_miou)
        print("Validation mAP@50: ", val_mAP50)
        print("Validation mAP: ", val_mAP)
    else:
        print(f"{args.model} is not supported")
    