import os
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.v2 as T

from models.model import build_model
from datasets.dataset import CityscapesDataset, get_dataloader
from utils.loss_fn import DiceFocalCELoss
from utils.visualize import convert_instance_to_semantic, show_loss_graph, show_lr_graph
from configs.config import config, id_to_trainId_map_20c, id_to_trainId_map_19c, id_to_trainId_map_9c, id_to_trainId_map_8c

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


def train_maskrcnn(train_type, config):
    train_augmentation_maskrcnn = T.Compose([
        T.Resize((512,1024)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.RandomSolarize(threshold=19.0),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    if train_type == "full":
        id_to_trainId_map = id_to_trainId_map_20c
        num_classes = 20
    else:
        id_to_trainId_map = id_to_trainId_map_9c
        num_classes = 9

    model = build_model(config, num_classes=num_classes)
    train_dataset = CityscapesDataset(config, id_to_trainId_map, ignore_index=0, transform=train_augmentation_maskrcnn, split='train')
    model, _ = train_maskrcnn_semantic(model, train_dataset, config)

def train_unet(train_type, config):
    train_augmentation_unet = T.Compose([
        T.Resize((512,1024)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.RandomSolarize(threshold=19.0),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    if train_type == "full":
        id_to_trainId_map = id_to_trainId_map_19c
        num_classes = 19
    else:
        id_to_trainId_map = id_to_trainId_map_8c
        num_classes = 8

    model = build_model(config, num_classes=num_classes)
    train_dataset = CityscapesDataset(config, id_to_trainId_map, ignore_index=255, transform=train_augmentation_unet, split='train')
    model, _ = train_unet_semantic(model, train_dataset, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training model and autosave to checkpoint path")
    parser.add_argument("-m", "--model", choices=["maskrcnn", "unet"], type=str, help="Model name (MaskRCNN or UNet)")
    parser.add_argument("-t", "--type", choices=["full", "pedveh"], type=str, help="Full or person+vehicle")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints", help="Path to save model after training")
    args = parser.parse_args()
    
    config.model_name = args.model
    config.ckpt_path = args.ckpt_path
    if args.model == "maskrcnn":
        train_maskrcnn(args.type, config)
    elif args.model == "unet":
        train_unet(args.type, config)
    else:
        print(f"{args.model} is not supported")