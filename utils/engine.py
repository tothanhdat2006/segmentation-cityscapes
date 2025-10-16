import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from datasets import get_dataloader
from utils import DiceFocalCELoss, FocalLoss, convert_instance_to_semantic, show_loss_graph, show_lr_graph, compute_ap_range

def train_maskrcnn_semantic(model, train_dataset, config):
    n_total = len(train_dataset) * config.n_epochs
    model.to(config.device)
    model.train()
    train_dataloader = get_dataloader(train_dataset, config)
    optimizer = optim.AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps, weight_decay=config.optimizer.params.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=5, cooldown=3, min_lr=1e-5)

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total, eta_min=1e-6)

    scaler = None
    if config.scaler:
        scaler = torch.amp.GradScaler(config.device, enabled=True)
    step_losses = []
    lrs = []
    with tqdm(total=n_total, unit="step") as pbar:
        for epoch in range(1, config.n_epochs+1):
            epoch_loss = 0.0
            for batch in train_dataloader:
                images, targets = batch
                
                images = [image.to(config.device) for image in images]
                targets = [{k: v.to(config.device) for k, v in t.items()} for t in targets]
                with torch.amp.autocast(device_type=config.device, enabled=(scaler is not None)):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                loss_value = losses
        
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss_value).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_value.backward()
                    optimizer.step()
                    
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_metric = loss_value.item()
                    scheduler.step(scheduler_metric)
                else:
                    scheduler.step()

                
                step_losses.append(loss_value.item())
                lrs.append(scheduler.get_last_lr()[0])
                epoch_loss += loss_value.item()
                pbar.update(len(images))
                pbar.set_postfix(
                    current_loss=f"{loss_value.item():.4f}",
                    avg_loss=f"{sum(step_losses)/len(step_losses):.4f}", 
                    lr=f"{scheduler.get_last_lr()[0]:.1e}"
                )
                
            epoch_loss /= len(train_dataloader)
            print(f"Epoch {epoch}/{config.n_epochs+1} average loss: {epoch_loss}")

    torch.save({
        'epoch': config.n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step_losses': step_losses,
    }, os.path.join(config.ckpt_path, f"maskrcnn_ckpt_{config.n_epochs}epoch.pth"))

    show_loss_graph(step_losses)
    show_lr_graph(lrs)
    return model, step_losses


def train_unet_semantic(model, train_dataset, config):
    n_total = len(train_dataset) * config.n_epochs
    model.to(config.device)
    model.train()

    train_dataloader = get_dataloader(train_dataset, config)
    optimizer = optim.AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps, weight_decay=config.optimizer.params.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=4, cooldown=2, min_lr=1e-5)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total, eta_min=1e-5)

    combined_loss = DiceFocalCELoss([0.8, 1.0, 1.0], ignore_index=255, alpha=0.25, gamma=2)
    scaler = None
    if config.scaler:
        scaler = torch.amp.GradScaler(config.device, enabled=True)
    step_losses = []
    lrs = []
    with tqdm(total=n_total, unit="step") as pbar:
        for epoch in range(1, config.n_epochs+1):
            epoch_loss = 0.0
            for batch in train_dataloader:
                images, targets = batch
                images = torch.stack(images).to(config.device)
                semantic_masks = [convert_instance_to_semantic(t['masks'], t['labels']) for t in targets]
                semantic_masks = torch.stack(semantic_masks).squeeze(1).long() 
                
                onehot_masks = torch.zeros((semantic_masks.shape[0], model.n_classes, semantic_masks.shape[1], semantic_masks.shape[2]), dtype=torch.long)
                for i in range(semantic_masks.shape[0]):
                    unique_classes_in_this_mask = torch.unique(semantic_masks[i])
                    for class_id in range(model.n_classes):
                        if class_id in unique_classes_in_this_mask:
                            onehot_masks[i, class_id, :, :] = (semantic_masks[i] == class_id).long()

                with torch.amp.autocast(device_type=config.device, enabled=(scaler is not None)):
                    predicted_masks = model(images) # 
                    predicted_masks = predicted_masks.cpu().float()
                    losses = combined_loss(predicted_masks, onehot_masks, semantic_masks) 
                    loss_value = losses['ce'] + losses['dl'] + losses['fc']
                    
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss_value).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    optimizer.step()
                    
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_metric = loss_value.item()
                    scheduler.step(scheduler_metric)
                else:
                    scheduler.step()

                step_losses.append(loss_value.half().item())
                epoch_loss += loss_value.item()
                pbar.update(len(images))
                pbar.set_postfix(
                    current_loss=f"{loss_value.item():.4f}",
                    avg_loss=f"{sum(step_losses)/len(step_losses):.4f}", 
                    ce=f"{losses['ce'].item():.4f}",
                    dice=f"{losses['dl'].item():.4f}",
                    focal=f"{losses['fc'].item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.1e}"
                )

            epoch_loss /= len(train_dataloader)
            print(f"Epoch {epoch}/{config.n_epochs+1} loss: {epoch_loss}")
            
    torch.save({
        'epoch': config.n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step_losses': step_losses,
    }, os.path.join(config.ckpt_path, f"unet_ckpt_{config.n_epochs}epoch.pth"))
    show_loss_graph(step_losses)
    show_lr_graph(lrs)
    return model, step_losses

def train_deeplab_semantic(model, train_dataset, config):
    n_total = len(train_dataset) * config.n_epochs
    model.to(config.device)
    model.train()

    train_dataloader = get_dataloader(train_dataset, config, is_train=True, collate_fn=None)
    optimizer = optim.AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps, weight_decay=config.optimizer.params.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=20, cooldown=5, min_lr=5e-5)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_total, eta_min=1e-5)
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.backbone.parameters(), 'lr': 0.1 * config.optimizer.params.lr},
    #     {'params': model.classifier.parameters(), 'lr': config.optimizer.params.lr},
    # ], lr=config.optimizer.params.lr, momentum=0.9, weight_decay=config.optimizer.params.weight_decay)
    # scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=n_total, power=0.9)
    if config.loss_fn == "cross_entropy":
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif config.loss_fn == "focal_loss":
        criterion = FocalLoss(ignore_index=255, reduction='mean')

    scaler = None
    if config.scaler:
        scaler = torch.amp.GradScaler(config.device, enabled=True)
    step_losses = []
    lrs = []
    
    iter_times = []
    mem_usages = []
    
    total_train_start_time = time.time()
    
    with tqdm(total=n_total, unit="step") as pbar:
        for epoch in range(1, config.n_epochs+1):
            epoch_loss = 0.0
            for batch in train_dataloader:
                iter_start_time = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(config.device)
                    
                images, targets = batch
                images = images.to(config.device, dtype=torch.float32)
                targets = targets.to(config.device, dtype=torch.long)
                
                with torch.amp.autocast(device_type=config.device, enabled=(scaler is not None)):
                    predicted_logits = model(images)
                    loss_value = criterion(predicted_logits, targets.squeeze(1))
                    
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss_value).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_value.backward()
                    optimizer.step()
                    
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_metric = loss_value.item()
                    scheduler.step(scheduler_metric)
                else:
                    scheduler.step()

                step_losses.append(loss_value.half().item())
                lrs.append(scheduler.get_last_lr()[0])
                epoch_loss += loss_value.item()

                train_time_s_iter = time.time() - iter_start_time
                iter_times.append(train_time_s_iter)
                if torch.cuda.is_available():
                    train_mem_gb = torch.cuda.max_memory_allocated(config.device) / (1024 ** 3)
                    mem_usages.append(train_mem_gb)
                else:
                    train_mem_gb = 0.0
                
                pbar.update(len(images))
                pbar.set_postfix(
                    current_loss=f"{loss_value.item():.4f}",
                    avg_loss=f"{sum(step_losses)/len(step_losses):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.1e}",
                    mem_gb=f"{train_mem_gb:.2f}",
                    time_s_iter=f"{train_time_s_iter:.2f}"
                )


            epoch_loss /= len(train_dataloader)
            print(f"Epoch {epoch}/{config.n_epochs+1} loss: {epoch_loss}")

    total_train_time_s = time.time() - total_train_start_time
    total_train_time_hr = total_train_time_s / 3600
    avg_train_time_s_iter = sum(iter_times) / len(iter_times)
    avg_train_mem_gb = sum(mem_usages) / len(mem_usages) if len(mem_usages) > 0 else 0

    print("\n--- Measuring Inference Speed ---")
    model.eval()
    total_inference_model_times = []
    total_inference_total_times = []
    num_inference_samples = 100
    num_warmup_samples = 5

    with torch.no_grad():
        print(f"Performing {num_warmup_samples} warm-up runs...")
        for i in range(num_warmup_samples):
            image_for_inference, _ = train_dataset[i]
            image_tensor = image_for_inference.to(config.device, dtype=torch.float32).unsqueeze(0)
            _ = model(image_tensor)

        print(f"Measuring over {num_inference_samples} samples...")
        for i in range(num_inference_samples):
            image_for_inference, _ = train_dataset[i]

            inf_total_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inf_model_start_time = time.time()

            image_tensor = image_for_inference.to(config.device, dtype=torch.float32).unsqueeze(0)
            _ = model(image_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inf_model_end_time = time.time()
            inf_total_end_time = time.time()

            total_inference_model_times.append(inf_model_end_time - inf_model_start_time)
            total_inference_total_times.append(inf_total_end_time - inf_total_start_time)

    avg_inference_model_time_s_im = np.mean(total_inference_model_times)
    avg_inference_total_time_s_im = np.mean(total_inference_total_times)


    print("\n--- Training & Inference Summary ---")
    print(f"| train mem (GB)             | {avg_train_mem_gb:.2f}")
    print(f"| train time (s/iter)        | {avg_train_time_s_iter:.4f}")
    print(f"| total train time (hr)      | {total_train_time_hr:.4f}")
    print(f"| inference model time (s/im)| {avg_inference_model_time_s_im:.4f}")
    print(f"| inference total time (s/im)| {avg_inference_total_time_s_im:.4f}")
    print("------------------------------------\n")
    
    torch.save({
        'epoch': config.n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step_losses': step_losses,
    }, os.path.join(config.ckpt_path, f"{config.model_name}_ckpt_{config.n_epochs}epoch.pth"))
    show_loss_graph(step_losses)
    show_lr_graph(lrs)
    return model, step_losses

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
                    if targets[i]['labels'].shape[0] == 1 and targets[i]['labels'][0] == 0: # exist image without person or vehicle
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
                    if targets[i]['labels'].shape[0] == 1 and targets[i]['labels'][0] == 0: # exist image without person or vehicle
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
    print('Evaluating U-Net model with CustomMeanIoU...')
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
                    if targets[i]['labels'].shape[0] == 1 and targets[i]['labels'][0] == 255: # exist image without person or vehicle
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
                    if targets[i]['labels'].shape[0] == 1 and targets[i]['labels'][0] == 255: # exist image without person or vehicle
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

def evaluate_deeplab_model(model, dataloader, device, num_classes, mIoU_metric):
    """
    Correctly evaluates a DeepLabv3 model using CustomMeanIoU, ignoring the class index 255.
    """
    print('Evaluating DeepLabv3 model with CustomMeanIoU...')
    model.eval()

    cnt = 0
    mAP = 0.0
    mAP50 = 0.0
    
    with torch.no_grad():
        with tqdm(desc="Evaluating DeepLabv3", total=len(dataloader), leave=False) as pbar:
            for images, targets in dataloader:
                images = images.to(device, dtype=torch.float32) # [B, 3, H, W]
                semantic_masks = targets.squeeze(1)
                cnt += len(images)     
                
                predicted_masks = model(images) # [B, 19, H, W]
                
                # ===================================================== Calculate mIoU ====================================================
                onehot_masks = torch.zeros((semantic_masks.shape[0], num_classes, semantic_masks.shape[1], semantic_masks.shape[2]), dtype=torch.long)
                for i in range(semantic_masks.shape[0]):
                    unique_classes_in_this_mask = torch.unique(semantic_masks[i])
                    for class_id in range(num_classes):
                        if class_id in unique_classes_in_this_mask:
                            onehot_masks[i, class_id, :, :] = (semantic_masks[i] == class_id).long()

                mIoU_metric.update(predicted_masks.long().cpu(), onehot_masks.long().cpu())

                # ===================================================== Calculate mAP =====================================================
                dummy_boxes = np.zeros((num_classes, 4)) # dummy mAP doesnt process what inside
                dummy_scores = np.full(num_classes, 1.0) # assume model is sure about the mask
                pred_labels = np.arange(num_classes)
                for i in range(len(predicted_masks)):
                    # target_mask = targets[i] # (1, H, W)
                    onehot_partial_masks = []
                    target_labels = np.unique(targets[i])[:-1]
                    for class_id in target_labels:
                        onehot_partial_masks.append((semantic_masks[i] == class_id).long())
                    onehot_partial_masks = np.array(onehot_partial_masks).transpose(1, 2, 0)
                    pred_masks = (predicted_masks[i] > 0.5).to('cpu').permute(1, 2, 0)
                    pred_masks = np.array(pred_masks)
                    # print(target_labels)
                    # print(target_labels.shape, onehot_partial_masks.shape)
                    
                    mAP50 += compute_ap_range(dummy_boxes, target_labels, onehot_partial_masks,
                                        dummy_boxes, pred_labels, dummy_scores, pred_masks,
                                        iou_thresholds=[0.5], verbose=0)
                    
                    mAP += compute_ap_range(dummy_boxes, target_labels, onehot_partial_masks,
                                        dummy_boxes, pred_labels, dummy_scores, pred_masks,
                                        iou_thresholds=None, verbose=0)
                
                
                pbar.update(1)
                pbar.set_postfix(iou=f"{mIoU_metric.compute():.4f}", mAP50=f"{mAP50/cnt:.4f}", mAP=f"{mAP/cnt:.4f}")
                # if cnt >= 8:
                #     break

    val_miou = mIoU_metric.compute()
    mAP50 /= cnt
    mAP /= cnt
    model.train()
    return val_miou, mAP50, mAP