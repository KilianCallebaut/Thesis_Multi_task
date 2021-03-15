from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
from sklearn.metrics import f1_score

plt.ion()  # interactive mode

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# https://github.com/sugi-chan/pytorch_multitask/blob/master/pytorch%20multi-task-Copy2.ipynb
class fgo_dataset(Dataset):
    def __init__(self, king_of_lists, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.king_of_lists = king_of_lists
        self.transform = transform

    def __getitem__(self, index):
        # random_index = randrange(len(self.king_of_lists[index]))

        img1 = Image.open('Datasets\\images/' + self.king_of_lists[0][index])
        img1 = img1.convert('RGB')

        gender = self.king_of_lists[1][index]  # gender
        region = self.king_of_lists[2][index]  # region
        fight = self.king_of_lists[3][index]  # fighting
        alignment = self.king_of_lists[4][index]  # alignment
        color = self.king_of_lists[5][index]  # color
        if self.transform is not None:
            img1 = self.transform(img1)
        list_of_labels = [torch.from_numpy(np.array(gender)),
                          torch.from_numpy(np.array(region)),
                          torch.from_numpy(np.array(fight)),
                          torch.from_numpy(np.array(alignment)),
                          torch.from_numpy(np.array(color))]
        # list_of_labels = torch.FloatTensor(list_of_labels)

        # print(img1.shape,len(list_of_labels),
        #     len(gender),
        #     len(region),
        #     len(fight),
        #     len(alignment),
        #    len(color))

        # torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        print("=================================================")
        print(list_of_labels)
        print("=================================================")

        return img1, list_of_labels[0], list_of_labels[1], list_of_labels[2], list_of_labels[3], list_of_labels[4]

    def __len__(self):
        return len(self.king_of_lists[0])
