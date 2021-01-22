import os
import glob
import random

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from xml.etree import ElementTree as ET
from xml.dom import minidom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class PenaltyDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, gt_path, device, transform=False):
        self.images_path = images_path
        self.gt_path = gt_path
        self.filenames = list(map(os.path.basename, glob.glob(os.path.join(images_path, "*.png"))))
        self.transform = transform
        self.feed_shape = [750, 1333]
        self.device = device

        self.class_names = ['player', 'referee', 'ball']

    def read_xml(self, idx):
        filename = self.filenames[idx].rsplit('.', 1)[0] + '.xml'
        tree = ET.parse(os.path.join(self.gt_path, filename))
        root = tree.getroot()

        bboxes = []
        labels = []
        for obj in root.findall(".//object"):
            label = self.class_names.index(obj[0].text)
            for elem in obj.findall(".//bndbox"):
                x0 = int(elem[0].text)
                y0 = int(elem[1].text)
                x1 = int(elem[2].text)
                y1 = int(elem[3].text)
                bboxes.append([x0, y0, x1, y1])
                labels.append(label)

        return bboxes, labels

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        image = Image.open(os.path.join(self.images_path, filename)).convert("RGB")
        
        target = {}
        if self.gt_path is not None:
            mask = Image.open(os.path.join(self.gt_path, filename)).convert("RGB")
            boxes, labels = self.read_xml(idx)

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            image_id = torch.tensor([idx])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target["boxes"] = boxes.to(self.device)
            target["labels"] = labels.to(self.device)
            target["image_id"] = image_id.to(self.device)
            target["iscrowd"] = iscrowd.to(self.device)
            target["area"] = area.to(self.device)
            target["seg_mask"] = None
        # else:
        #     mask, boxes, labels = None, None, None

        # sample = {"image": image, "mask": mask}
        if self.transform:
            if random.random() > 0.:
                image = TF.hflip(image)

                target['boxes'][:, 0] = image.size[0] - target['boxes'][:, 0]
                target['boxes'][:, 2] = image.size[0] - target['boxes'][:, 2]

                target['boxes'][:, [0, 2]] = target['boxes'][:, [2, 0]]
                if self.gt_path is not None:
                    mask  = TF.hflip(mask)

        rw = self.feed_shape[1] / image.size[0]
        rh = self.feed_shape[0] / image.size[1]

        target['boxes'][:, 0] *= rw
        target['boxes'][:, 1] *= rh
        target['boxes'][:, 2] *= rw
        target['boxes'][:, 3] *= rh
  
        image = TF.resize(image, self.feed_shape, Image.BILINEAR)

        # img1 = ImageDraw.Draw(image)   
        # for bbox in target['boxes']:
        #     x1, y1, x2, y2 = bbox
        #     print(bbox)
        #     img1.rectangle([(x1, y1), (x2, y2)], outline ="red") 
        # image.show() 
        # exit(0)
        # plt.imshow(image)

        image = TF.to_tensor(image).to(self.device)

        if self.gt_path is not None:
            mask = TF.resize(mask, self.feed_shape, Image.NEAREST)
            mask = np.array(mask)
            mask_labels = np.zeros(mask.shape[:2], dtype=np.uint8)

            mask_labels[mask[:, :, 0] == 255] = 0
            mask_labels[mask[:, :, 1] == 255] = 1
            mask_labels[mask[:, :, 2] == 255] = 2
            mask = torch.LongTensor(mask_labels)[None].to(self.device)
            target['seg_mask'] = mask
        return image, target
        
    def __len__(self):
        return len(self.filenames)
