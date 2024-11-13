# import config
import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from utils import (
    iou_width_height as iou,
    non_max_suppression as nms,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.S = S
        self.C = C
        self.transform = transform
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # x, y, w, h -> w, h, x, y 
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes) 
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # 6 -> (pc, x, y, w, h, class) 

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) 
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) 
            x, y, w, h, class_label = box 
            has_anchor = [False] * 3 # each scale should have one anchor

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale # 0, 1, 2; which scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # 0, 1, 2; which anchor in scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x) # which cell in the grid, eg. x=0.4, y=0.4 -> i=5, j=5
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i # eg. x=0.4, y=0.4, S=13 -> x_cell=0.4-5=0.4, y_cell=0.4-5=0.4; x_cell, y_cell is the relative position of the box in the cell; between 0 and 1
                    width_cell, height_cell = (
                        w * S, # eg. w=0.2, S=13 -> width_cell=0.2*13=2.6
                        h * S,
                    )
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates # x, y, w, h
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) # class
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh: # if the anchor is not taken and the iou is greater than the threshold, then it's a false positive
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction

        return image, tuple(targets)






