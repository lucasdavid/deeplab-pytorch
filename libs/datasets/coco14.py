#!/usr/bin/env python
# coding: utf-8
#
# Author:   Zhaozheng
# Created:  2021-10

from __future__ import absolute_import, print_function

import os
from glob import glob

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset


class COCO14(_BaseDataset):

    """Flag indicating that we should expect segmentation masks to be placed under
    short IDs `{ID}.png` instead of complete ones `COCO_train2014_{ID}.png`.
    E.g., data/datasets/coco14/SegmentationClass/000000103969.png
    """
    SHORT_LABEL_IDS = True

    def __init__(self, **kwargs):
        super(COCO14, self).__init__(**kwargs)

    def _set_files(self):
        # Create data list via {train, test, all}.txt
        if self.split in ["train", "val"]:
            file_list = os.path.join(self.root, self.split + ".txt")
            with open(file_list, "r") as file_list:
              ids = []
              for entry in file_list:
                image_path = entry.strip().split()[0]
                image_id = os.path.split(image_path[:-4])[-1]
                ids.append(image_id)
            self.files = ids
            print("loaded files:", self.files[:10])
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = label_id = self.files[index]
        if self.SHORT_LABEL_IDS:
            label_id = image_id.split("_")[-1]
        image_path = os.path.join(self.root, "JPEGImages", image_id + ".jpg")
        label_path = os.path.join(self.root, "SegmentationClass", label_id + ".png")
        # if 'MSCOCO2' in self.label_dir:
        #     label_path = os.path.join(self.label_dir, image_id + ".png")
        # else:
        #     label_path = os.path.join(self.label_dir, 'COCO_train2014_'+image_id + ".png")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)

        return image_id, image, label
