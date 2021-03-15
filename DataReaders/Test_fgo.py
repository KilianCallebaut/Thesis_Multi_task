from __future__ import print_function, division

from DataReaders.DataReader import DataReader
import torch
from torch import nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image

from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from MultiTask.HardSharing import HardSharing
from MultiTask.pytorch_multitask_example.multi_output_model import multi_output_model
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tasks.Task import Task
from Tasks.TaskDataset import TaskDataset
from Training.Training import Training

plt.ion()  # interactive mode
import pandas as pd
from sklearn.model_selection import train_test_split


class Test_fgo(DataReader):
    def __init__(self):
        dat = pd.read_csv('..\\Datasets\\fgo_multiclass_labels.csv')
        X = dat['image_name']
        y = dat[['white', 'red',
                 'green', 'black', 'blue', 'purple', 'gold', 'silver', 'gender_Female',
                 'gender_Male', 'region_Asia', 'region_Egypt', 'region_Europe',
                 'region_Middle East', 'fighting_type_magic', 'fighting_type_melee',
                 'fighting_type_ranged', 'alignment_CE', 'alignment_CG', 'alignment_CN',
                 'alignment_LE', 'alignment_LG', 'alignment_LN', 'alignment_NE',
                 'alignment_NG', 'alignment_TN']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        X_train = X_train.values.tolist()
        X_test = X_test.values.tolist()

        # train
        colors = ['white', 'red',
                  'green', 'black', 'blue', 'purple', 'gold', 'silver']
        color_train = y_train[colors]
        color_nodes = color_train.shape[1]
        color_train = color_train.values.tolist()

        gender = ['gender_Female',
                  'gender_Male']
        gender_train = y_train[gender]
        gender_nodes = gender_train.shape[1]
        gender_train = gender_train.values.tolist()

        region = ['region_Asia', 'region_Egypt', 'region_Europe',
                  'region_Middle East']
        region_train = y_train[region]
        region_nodes = region_train.shape[1]
        region_train = region_train.values.tolist()

        fighting_style = ['fighting_type_magic', 'fighting_type_melee',
                          'fighting_type_ranged']
        fighting_style_train = y_train[fighting_style]
        fighting_nodes = fighting_style_train.shape[1]
        fighting_style_train = fighting_style_train.values.tolist()

        alignment = ['alignment_CE', 'alignment_CG', 'alignment_CN',
                     'alignment_LE', 'alignment_LG', 'alignment_LN', 'alignment_NE',
                     'alignment_NG', 'alignment_TN']
        alignment_train = y_train[alignment]
        alignment_nodes = alignment_train.shape[1]
        alignment_train = alignment_train.values.tolist()

        print(color_nodes, gender_nodes, region_nodes, fighting_nodes, alignment_nodes)
        # test
        colors = ['white', 'red',
                  'green', 'black', 'blue', 'purple', 'gold', 'silver']
        color_test = y_test[colors]
        color_nodes = color_test.shape[1]
        color_test = color_test.values.tolist()

        gender = ['gender_Female',
                  'gender_Male']
        gender_test = y_test[gender]
        gender_nodes = gender_test.shape[1]
        gender_test = gender_test.values.tolist()

        region = ['region_Asia', 'region_Egypt', 'region_Europe',
                  'region_Middle East']
        region_test = y_test[region]
        region_nodes = region_test.shape[1]
        region_test = region_test.values.tolist()

        fighting_style = ['fighting_type_magic', 'fighting_type_melee',
                          'fighting_type_ranged']
        fighting_style_test = y_test[fighting_style]
        fighting_nodes = fighting_style_test.shape[1]
        fighting_style_test = fighting_style_test.values.tolist()

        alignment = ['alignment_CE', 'alignment_CG', 'alignment_CN',
                     'alignment_LE', 'alignment_LG', 'alignment_LN', 'alignment_NE',
                     'alignment_NG', 'alignment_TN']
        alignment_test = y_test[alignment]
        alignment_nodes = alignment_test.shape[1]
        alignment_test = alignment_test.values.tolist()

        print(color_nodes, gender_nodes, region_nodes, fighting_nodes, alignment_nodes)

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # own model
        # Important: the tasks in task_list has to be the same as the order in concatTaskDataset
        # Required:
        # ConcatTaskDataset
        # ->    list of TaskDataset
        # ->    inputs: list of size nr_instances, flatened tensors
        # ->    targets: tensors of right outputs
        # ->    name: name as task, consistent with task_list
        # X_input_size: input nodes amount
        # task_list: list of object Task:
        # ->    name: name as task, consistent with concattaskdataset
        # ->    output_labels: list of string with output labels

        X_loaded = [Image.open('../Datasets/images/' + x) for x in X_train]
        X_loaded = [x.convert('RGB') for x in X_loaded]
        X_loaded = [data_transforms['train'](x) for x in X_loaded]
        X_loaded = [torch.flatten(x, 0, -1) for x in X_loaded]
        self.input_nodes = X_loaded[0].shape[0]

        self.training_dataset = ConcatTaskDataset(
            [TaskDataset(inputs=X_loaded, targets=gender_train, name='gender', labels=gender),
             TaskDataset(inputs=X_loaded, targets=region_train, name='region', labels=region),
             TaskDataset(inputs=X_loaded, targets=fighting_style_train, name='fighting', labels=fighting_style),
             TaskDataset(inputs=X_loaded, targets=alignment_train, name='alignment', labels=alignment),
             TaskDataset(inputs=X_loaded, targets=color_train, name='colors', labels=colors)]
        )
        self.task_list = self.training_dataset.get_task_list()

    def toTaskDataset(self):
        return self.training_dataset

    def get_task_list(self):
        return self.task_list

    def get_input_nodes(self):
        return self.input_nodes
