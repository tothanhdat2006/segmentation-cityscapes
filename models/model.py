import torch
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

from models.unet import UNET
from torchvision.models import resnet as Resnet
from models.deeplabv3 import DeepLabV3, DeepLabHead, DeepLabHeadV3Plus, mobilenetv2
from models.deeplabv3.utils import IntermediateLayerGetter
    
def deeplabv3_resnet(deeplab_name, backbone_name, n_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = Resnet.__dict__[backbone_name](
        pretrained = pretrained_backbone,
        replace_stride_with_dilation = replace_stride_with_dilation
    )
    inplanes = 2048
    low_level_planes = 256
    if deeplab_name == "deeplabv3":
        return_layers = {"layer4": "out"}
        classifier = DeepLabHead(inplanes, n_classes, aspp_dilate)
    elif deeplab_name == "deeplabv3plus":
        return_layers = {"layer4": "out", "layer1": "low_level"}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, n_classes, aspp_dilate)
        
    backbone = IntermediateLayerGetter(backbone, return_layers = return_layers)
    model = DeepLabV3(backbone, classifier)
    return model

def deeplabv3_mobilenet(deeplab_name, backbone_name, n_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained = pretrained_backbone, output_stride = output_stride)
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    if deeplab_name == "deeplabv3":
        return_layers = {"high_level_features": "out"}
        classifier = DeepLabHead(inplanes, n_classes, aspp_dilate)
    elif deeplab_name == "deeplabv3plus":
        return_layers = {"high_level_features": "out", "low_level_features": "low_level"}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, n_classes, aspp_dilate)
        
    backbone = IntermediateLayerGetter(backbone, return_layers = return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def build_MaskRCNN(num_classes):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    maskrcnn = maskrcnn_resnet50_fpn_v2(weights)
    
    in_features_box = maskrcnn.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
    
    dim_reduced = maskrcnn.roi_heads.mask_predictor.conv5_mask.out_channels
    
    maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=num_classes)
    maskrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=num_classes)
    return maskrcnn

def build_DeepLabv3(full_name, num_classes=19, output_stride=8, pretrained_backbone=True):
    deeplab_name, backbone_name = full_name.split("_")
    
    assert deeplab_name == "deeplabv3" or deeplab_name == "deeplabv3plus", "Supported model: deeplabv3, deeplabv3plus"
    assert backbone_name.startswith("resnet") or backbone_name.startswith("mobilenet"), "Supported backbone: mobilenet, resnet50, resnet101"
    if backbone_name.startswith("resnet"):
        assert backbone_name == "resnet50" or backbone_name == "resnet101", "Supported backbone: mobilenet, resnet50, resnet101"
        return deeplabv3_resnet(deeplab_name, backbone_name, num_classes, output_stride, pretrained_backbone)
    else:
        return deeplabv3_mobilenet(deeplab_name, backbone_name, num_classes, output_stride, pretrained_backbone)

def build_model(model_config, num_channels=3, num_classes=20, output_stride=8):
    if model_config.model_name.lower() == "unet":
        return UNET(num_channels, num_classes)
    elif model_config.model_name.lower() == "maskrcnn":
        return build_MaskRCNN(num_classes)
    elif model_config.model_name.lower().startswith("deeplabv3"):
        assert "_" in model_config.model_name.lower(), "A backbone should be specified for deeplabv3"
        return build_DeepLabv3(model_config.model_name.lower(), num_classes, output_stride, True)
    else:
        print(f"{model_config.model_name} is currently not supported")

def load_maskrcnn(num_classes, model_ckpt_path):
    model = build_MaskRCNN(num_classes=num_classes)
    weight = torch.load(model_ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(weight['model_state_dict'])
    return model

def load_unet(num_classes, model_ckpt_path):
    model = UNET(3, num_classes=num_classes)
    weight = torch.load(model_ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(weight['model_state_dict'])
    return model

def load_deeplab(model_name, num_classes, model_ckpt_path, output_stride=8):
    model = build_DeepLabv3(model_name, num_classes=num_classes, output_stride=output_stride, pretrained_backbone=False)
    weight = torch.load(model_ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(weight['model_state_dict'])
    return model