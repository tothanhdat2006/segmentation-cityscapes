import argparse

import torch
import torchvision.transforms.v2 as T
from torchmetrics.segmentation import MeanIoU

from models import load_maskrcnn, load_unet, load_deeplab
from utils import CustomMeanIoU, evaluate_maskrcnn_model, evaluate_unet_model, evaluate_deeplab_model
from datasets import CityscapesDataset, get_dataloader
from configs.config import config, id_to_trainId_map_20c, id_to_trainId_map_19c, id_to_trainId_map_9c, id_to_trainId_map_8c

def validate_maskrcnn(args, config):
    val_augmentation_maskrcnn = T.Compose([
        T.Resize((512,1024)) if args.resolution == "512" else T.Resize((800,1024)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if args.type == "full":
        mIoU_metric = MeanIoU(num_classes=20, include_background=False)
        id_to_trainId_map = id_to_trainId_map_20c
        num_classes = 20
    else:
        mIoU_metric = MeanIoU(num_classes=9, include_background=False)
        id_to_trainId_map = id_to_trainId_map_9c
        num_classes = 9
    
    model = load_maskrcnn(num_classes, args.ckpt_path).to(config.device)
    val_dataset = CityscapesDataset(config, id_to_trainId_map, instance=True, ignore_index=0, transform=val_augmentation_maskrcnn, split='val')
    val_dataloader = get_dataloader(val_dataset, config, is_train=False)
    val_miou, val_mAP50, val_mAP = evaluate_maskrcnn_model(model, val_dataloader, config.device, num_classes, mIoU_metric)
    return val_miou, val_mAP50, val_mAP
        
def validate_unet(args, config):
    val_augmentation_unet = T.Compose([
        T.Resize((512,1024)) if args.resolution == "512" else T.Resize((800,1024)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if args.type == "full":
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
    
    model = load_unet(num_classes, args.ckpt_path).to(config.device)
    val_dataset = CityscapesDataset(config, id_to_trainId_map, instance=True, ignore_index=255, transform=val_augmentation_unet, split='val')
    val_dataloader = get_dataloader(val_dataset, config, is_train=False)
    val_miou, val_mAP50, val_mAP = evaluate_unet_model(model, val_dataloader, config.device, num_classes, mIoU_metric)
    return val_miou, val_mAP50, val_mAP

def validate_deeplab(args, config):
    val_augmentation_deeplab = T.Compose([
        T.Resize((512,1024)) if args.resolution == "512" else T.Resize((800,1024)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if args.type == "full":
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
    
    model = load_deeplab(num_classes, args.ckpt_path).to(config.device)
    val_dataset = CityscapesDataset(config, id_to_trainId_map, instance=False, ignore_index=255, transform=val_augmentation_deeplab, split='val')
    val_dataloader = get_dataloader(val_dataset, config, is_train=False)
    val_miou, val_mAP50, val_mAP = evaluate_deeplab_model(model, val_dataloader, config.device, num_classes, mIoU_metric)
    return val_miou, val_mAP50, val_mAP

if __name__ == "__main__":    
    parser = argparse.ArgumentParser("Validating the model performance")
    parser.add_argument("-m", "--model", type=str, help="Model name (maskrcnn, unet, deeplabv3-mobilenetv2, deeplabv3plus-mobilenetv2)")
    parser.add_argument("-t", "--type", default="full", choices=["full", "pedveh"], type=str, help="Full or person+vehicle")
    parser.add_argument("--resolution", default="512", choices=["512", "800"], type=str, help="Input resolution")
    parser.add_argument("--ckpt_path", type=str, help="Model checkpoint path")
    args = parser.parse_args()
    if args.model == "maskrcnn":
        val_miou, val_mAP50, val_mAP = validate_maskrcnn(args, config)
        print("Validation mIoU: ", val_miou)
        print("Validation mAP@50: ", val_mAP50)
        print("Validation mAP: ", val_mAP)
    elif args.model == "unet":
        val_miou, val_mAP50, val_mAP = validate_unet(args, config)
        print("Validation mIoU: ", val_miou)
        print("Validation mAP@50: ", val_mAP50)
        print("Validation mAP: ", val_mAP)
    else:
        print(f"{args.model} is not supported")
    