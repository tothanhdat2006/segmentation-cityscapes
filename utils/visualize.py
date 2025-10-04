import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import random
import colorsys

import torch

# Original: https://github.com/matterport/Mask_RCNN
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

# Original: https://github.com/matterport/Mask_RCNN
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

# Original: https://github.com/matterport/Mask_RCNN
def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=False,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors if not
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[class_ids[i]]

        # # Bounding box
        # if not np.any(boxes[i]):
        #     # Skip this instance. Has no bbox. Likely lost in image cropping.
        #     continue
        # y1, x1, y2, x2 = boxes[i]
        # if show_bbox:
        #     p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
        #                         alpha=0.7, linestyle="dashed",
        #                         edgecolor=color, facecolor='none')
        #     ax.add_patch(p)

        # # Label
        # if not captions:
        #     class_id = class_ids[i]
        #     score = scores[i] if scores is not None else None
        #     label = class_names[class_id]
        #     caption = "{} {:.3f}".format(label, score) if score else label
        # else:
        #     caption = captions[i]
        # ax.text(x1, y1 + 8, caption,
        #         color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        # padded_mask = np.zeros(
        #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.5)
        # for verts in contours:
        #     # Subtract the padding and flip (y, x) to (x, y)
        #     verts = np.fliplr(verts) - 1
        #     p = Polygon(verts, facecolor="none", edgecolor=color)
        #     ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()

def convert_instance_to_semantic(instance_masks, instance_labels):
    """
    Converts an instance mask of shape [N, H, W] to a semantic mask of shape [H, W].
    
    This is done by "painting" the class ID from instance_labels onto the
    corresponding binary mask from instance_masks.
    """
    if instance_masks.ndim != 3 or instance_labels.ndim != 1:
        raise ValueError(f"Expected shapes [N, H, W] and [N], but got "
                         f"{instance_masks.shape} and {instance_labels.shape}")
    
    # Reshape labels to [N, 1, 1] to allow broadcasting
    reshaped_labels = instance_labels.view(-1, 1, 1)
    semantic_mask = torch.max(instance_masks.float() * reshaped_labels, dim=0)[0]
    
    return semantic_mask.long()

def colorize_mask(mask, color_dict):
    """
    mask: HxW
    color_dict: {id: color tuple}
    Converts a HxW segmentation mask of class indices to a 3xHxW RGB image.
    """
    mask_np = mask.cpu().numpy()
    rgb_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    
    present_classes = np.unique(mask_np)
    for class_id in present_classes:
        if class_id in color_dict:
            indices = (mask_np == class_id)
            rgb_mask[indices] = color_dict[class_id]
            
    return torch.from_numpy(rgb_mask).permute(2, 0, 1)

def show_loss_graph(losses):
    plt.figure(figsize=(12,8))
    plt.title("Loss over training")
    plt.plot(losses, color='b', linewidth=2)
    
    plt.xlabel("steps")
    plt.ylabel("loss value")
    plt.xlim(0, len(losses)-1)
    
    plt.grid()
    plt.legend()
    plt.show()

def show_lr_graph(lrs):
    plt.figure(figsize=(12,8))
    plt.title("Learning rate over training")
    plt.plot(lrs, color='r', linewidth=2)
    
    plt.xlabel("steps")
    plt.ylabel("learning rate")
    plt.xlim(0, len(lrs)-1)
    
    plt.grid()
    plt.legend()
    plt.show()
    

def pred_probs_hist(model, viz_dataset, device, idx=None):
    if not idx:
        idx = np.random.randint(len(viz_dataset))
    img, _ = viz_dataset[idx]
    with torch.no_grad():
        images = img.unsqueeze(0).to(device)
        predictions = model(images)
            
        for i in range(len(predictions)):
            pred = predictions[i]
            pred_scores = pred['scores'].cpu().numpy()
            high_conf_indices = np.where(pred_scores >= 0.7)[0]
            pred_scores = pred_scores[high_conf_indices]
            pred_masks = (pred['masks'][high_conf_indices] > 0.5).squeeze(1).cpu().numpy()
            pred_labels = pred['labels'][high_conf_indices].cpu().numpy()
            # pred_boxes = pred['boxes'][high_conf_indices].cpu().numpy()

            ind = np.argsort(pred_labels, axis=0)
            pred_labels = pred_labels[ind]
            pred_masks = pred_masks[ind, ...]
            pred_masks = np.transpose(pred_masks, (1, 2, 0))
            plt.figure(figsize=(16, 8))
            num_rows = int(len(pred_labels)/3) + 1
            for i in range(int(len(pred_labels)/3)):
                if i*3+1 <= len(pred_labels):
                    plt.subplot(num_rows, 3, i*3+1)
                    p1 = pred_masks[..., i*3].copy().astype(float)
                    plt.hist(p1.reshape(-1))
                    plt.xlabel(f"Label: {pred_labels[i*3]}")

                if i*3+2 <= len(pred_labels):
                    plt.subplot(num_rows, 3, i*3+2)
                    p1 = pred_masks[..., i*3+1].copy().astype(float)
                    plt.hist(p1.reshape(-1))
                    plt.xlabel(f"Label: {pred_labels[i*3+1]}")

                if i*3+3 <= len(pred_labels):
                    plt.subplot(num_rows, 3, i*3+3)
                    p1 = pred_masks[..., i*3+2].copy().astype(float)
                    plt.hist(p1.reshape(-1))
                    plt.xlabel(f"Label: {pred_labels[i*3+2]}")
            plt.tight_layout()
            plt.show()