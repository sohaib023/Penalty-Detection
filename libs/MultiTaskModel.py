import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models._utils import IntermediateLayerGetter

from .transform import Transform
from .DummyBackbone import DummyBackbone

class MultiTaskModel(nn.Module):
    def __init__(self, training=True):
        super().__init__()

        return_layers = {
            "layer2": "0",
            "layer3": "1",
            "layer4": "2",
        }
        resnet50 = models.resnet50(pretrained=True, progress=True)
        self.backbone = IntermediateLayerGetter(resnet50, return_layers=return_layers)

        self.seg_dummy_backbone = DummyBackbone()
        self.segmentor = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=3)
        del self.segmentor.backbone
        self.segmentor.backbone = self.seg_dummy_backbone

        self.det_dummy_backbone = DummyBackbone()
        self.detector = models.detection.retinanet_resnet50_fpn(pretrained=False, progress=True, num_classes=3)
        del self.detector.backbone.body
        self.detector.backbone.body = self.det_dummy_backbone

        self.transform = Transform()

        self.seg_criterion = nn.CrossEntropyLoss()

        self.training = training

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise Exception("seg_masks must not be NoneType while MultiTaskModel.training is True.")

        seg_masks = [x['seg_mask'] for x in targets] if self.training and targets[0]['seg_mask'] is not None else None
        images, seg_masks, targets = self.transform(images, seg_masks, targets)

        out = self.backbone(images.tensors)
        self.seg_dummy_backbone.set_out_features({'out': out['2'], 'aux': out['1']})
        self.det_dummy_backbone.set_out_features(out)

        segmentations = self.segmentor(images.tensors)['out']
        det_output = self.detector(images.tensors, targets)
        # det_output = [{} for i in range(images.tensors.shape[0])]

        return_obj = {}

        if self.training:
            return_obj['cls_loss'] = det_output['classification']
            return_obj['bbox_loss'] = det_output['bbox_regression']
            return_obj['seg_loss'] = self.seg_criterion(segmentations, seg_masks.tensors.squeeze(1))
        else:
            return_obj = []
            for det, seg, size in zip(det_output, segmentations, images.image_sizes):
                seg = seg[:, :size[0], :size[1]]
                seg = torch.nn.functional.interpolate(seg[None], size=[750, 1333], mode='nearest')[0]
                det['seg'] = seg
                return_obj.append(det)

        return return_obj

