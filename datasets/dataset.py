import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
import torchvision.transforms.v2 as T

from cityscapesScripts.cityscapesscripts.helpers.labels import labels

class CityscapesDataset(Dataset):
    def __init__(self, config, id_to_trainId_map, ignore_index=255, transform=None, split='train'):
        super().__init__()
        self.images = [] # path to leftImg8bit
        self.label_images = [] # path to labels images

        self.ignore_index = ignore_index
        self.id_to_trainId_map = id_to_trainId_map
        self.config = config
        self._transform = transform

        self.class_names = [label.category for label in labels]

        left8bit_path = os.path.join(self.config.dataset_path, "leftImg8bit", split)
        for city in os.listdir(left8bit_path):
            city_images = list(os.listdir(os.path.join(left8bit_path, city)))
            city_images = [os.path.join(left8bit_path, city, image) for image in city_images]
            self.images.extend(city_images)
            
        for city in os.listdir(os.path.join(self.config.dataset_path, self.config.type, split)):
            city_dir = os.path.join(self.config.dataset_path, self.config.type, split, city)
            for file in os.listdir(city_dir):
                if file.endswith("labelIds.png"):
                    self.label_images.append(os.path.join(city_dir, file))

        self.images = sorted(self.images)
        self.label_images = sorted(self.label_images)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the raw RGB image
        img = read_image(str(self.images[idx]))

        img, target = self._get_semantic_item(idx, img)
        
        # Apply transformations (e.g., augmentations, resizing)
        if self._transform:
            img, target = self._transform(img, target)
            
        return img, target

    def _get_semantic_item(self, idx, img):
        mask = read_image(str(self.label_images[idx])).long() # Shape: [1, H, W]

        # Filter unlabel in mask
        converted_mask = torch.full_like(mask, self.ignore_index) 
        converted_mask[mask == -1] = self.ignore_index
        valid_pixels = (mask >= 0) & (mask <= 33)
        converted_mask[valid_pixels] = self.id_to_trainId_map[mask[valid_pixels]]

        obj_ids = torch.unique(converted_mask).to(dtype=torch.uint8)
        converted_mask = (converted_mask == obj_ids[:, None, None]).to(dtype=torch.long)
        boxes = masks_to_boxes(converted_mask)
        labels = obj_ids[:, None].to(dtype=torch.int64)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid_boxes_mask = (widths > 0) & (heights > 0)
        
        boxes = boxes[valid_boxes_mask]
        converted_mask = converted_mask[valid_boxes_mask]
        labels = labels[valid_boxes_mask]
        iscrowd = iscrowd[valid_boxes_mask] 

        # if 255 in labels:
        #     boxes = boxes[:-1, :]
        #     converted_mask = converted_mask[:-1, :, :]
        #     labels = labels[:-1]
        #     iscrowd = iscrowd[:-1] 
            
        target = {}
        target["masks"] = tv_tensors.Mask(converted_mask) # (N_instance, H, W)
        target["labels"] = labels.squeeze(1) # (N_instance)
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=T.functional.get_size(img)) # (N_instance, 4)
        target["iscrowd"] = iscrowd # (N_instance)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # (N_instance)
        return tv_tensors.Image(img), target

def collate_fn(batch):
    """
    Since the number of objects varies per image, we collate the images and targets
    into lists. The model expects a list of images and a list of targets dictionaries.
    """
    # batch is a list of tuples: [(img1, target1), (img2, target2), ...]
    images = [item[0] for item in batch] # This will be a list of image tensors/tv_tensors.Image
    targets = [item[1] for item in batch] # This will be a list of target dictionaries
    return images, targets

def get_dataloader(dataset, config, is_train=True):
    dataloader = DataLoader(dataset, batch_size=(config.per_gpu_train_batch_size if is_train else config.per_gpu_eval_batch_size), num_workers=config.num_workers, shuffle=is_train, collate_fn = collate_fn)
    return dataloader