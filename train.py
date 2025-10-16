import argparse
import numpy as np

import torch
import torchvision.transforms.v2 as T

from models import build_model
from datasets import CityscapesDataset
from utils import train_maskrcnn_semantic, train_unet_semantic, train_deeplab_semantic
from configs.config import config, id_to_trainId_map_20c, id_to_trainId_map_19c, id_to_trainId_map_9c, id_to_trainId_map_8c

from cityscapesScripts.cityscapesscripts.helpers.labels import labels


def train_maskrcnn(args, config):
    train_augmentation_maskrcnn = T.Compose([
        T.Resize((512,1024)) if args.resolution == "512" else T.Resize((800,1024)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.RandomSolarize(threshold=19.0),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    if args.type == "full":
        id_to_trainId_map = id_to_trainId_map_20c
        num_classes = 20
    else:
        id_to_trainId_map = id_to_trainId_map_9c
        num_classes = 9

    model = build_model(config, num_classes=num_classes)
    train_dataset = CityscapesDataset(config, id_to_trainId_map, instance=True, ignore_index=0, transform=train_augmentation_maskrcnn, split='train')
    model, _ = train_maskrcnn_semantic(model, train_dataset, config)

def train_unet(args, config):
    train_augmentation_unet = T.Compose([
        T.Resize((512,1024)) if args.resolution == "512" else T.Resize((800,1024)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.RandomSolarize(threshold=19.0),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    if args.type == "full":
        id_to_trainId_map = id_to_trainId_map_19c
        num_classes = 19
    else:
        id_to_trainId_map = id_to_trainId_map_8c
        num_classes = 8

    model = build_model(config, num_classes=num_classes)
    train_dataset = CityscapesDataset(config, id_to_trainId_map, instance=True, ignore_index=255, transform=train_augmentation_unet, split='train')
    model, _ = train_unet_semantic(model, train_dataset, config)

def train_deeplab(args, config):
    train_augmentation_deeplab = T.Compose([
        T.Resize((512,1024)) if args.resolution == "512" else T.Resize((800,1024)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.RandomSolarize(threshold=19.0),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    assert args.type == "full", "Currently only supports full 19 classes"
    id_to_trainId_map = np.array([label.trainId for label in labels])
    num_classes = 19

    config.optimizer.params.lr = 1e-3
    config.optimizer.params.weight_decay = 1e-4
    config.output_stride = 8
    config.per_gpu_train_batch_size = 4 # batch size must be multiple of 4
    config.loss_fn = "focal_loss"
    config.n_epochs = 8
    model = build_model(config, num_classes=num_classes)
    train_dataset = CityscapesDataset(config, id_to_trainId_map, instance=False, ignore_index=255, transform=train_augmentation_deeplab, split='train')
    model, _ = train_deeplab_semantic(model, train_dataset, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training model and autosave to checkpoint path")
    parser.add_argument("-m", "--model", type=str, help="Model name (maskrcnn, unet, deeplabv3-mobilenetv2, deeplabv3plus-mobilenetv2)")
    parser.add_argument("-t", "--type", default="full", choices=["full", "pedveh"], type=str, help="Full or person+vehicle")
    parser.add_argument("--resolution", default="512", choices=["512", "800"], type=str, help="Input resolution")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints", help="Path to save model after training")
    args = parser.parse_args()
    
    config.model_name = args.model
    config.ckpt_path = args.ckpt_path
    if args.model.lower() == "maskrcnn":
        train_maskrcnn(args, config)
    elif args.model.lower() == "unet":
        train_unet(args, config)
    elif args.model.lower().startswith("deeplabv3"):
        train_deeplab(args, config)
    else:
        print(f"{args.model} is not supported")