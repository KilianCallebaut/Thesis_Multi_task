import os
import random

import joblib
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from Tasks.Task import Task


class TaskDataset(Dataset):

    # data_lists: lists of lists of instances per task
    def __init__(
            self,
            inputs,
            targets,
            name,
            labels,
            output_module='softmax'  # 'sigmoid'
    ):
        self.inputs = inputs  # List of tensors
        self.targets = targets  # List of binary strings
        # self.name = name # String
        # self.labels = labels # List of strings
        self.task = Task(name, labels, output_module)
        self.pad_after = list()
        self.pad_before = list()

    def __getitem__(self, index):
        return self.inputs[index].float(), \
               torch.from_numpy(np.array(self.pad_before + self.targets[index] + self.pad_after)), \
               self.task.name

    def __len__(self):
        return len(self.inputs)

    def pad_targets(self, before: int, after: int):
        self.pad_before = [0 for _ in range(before)]
        self.pad_after = [0 for _ in range(after)]

    def save(self, base_path, extraction_method):
        joblib.dump(self.inputs, os.path.join(base_path, '{}_inputs.gz'.format(extraction_method)))
        t_s = torch.Tensor(self.targets)
        torch.save(t_s, os.path.join(base_path, 'targets.pt'))

        diction = {
            'task': self.task,
            'pad_after': self.pad_after,
            'pad_before': self.pad_before
        }
        joblib.dump(diction, os.path.join(base_path, 'other.obj'))

    def load(self, base_path, extraction_method):
        self.inputs = joblib.load(os.path.join(base_path, '{}_inputs.gz'.format(extraction_method)))
        t_l = torch.load(os.path.join(base_path, 'targets.pt'))
        self.targets = [[int(j) for j in i] for i in t_l]
        diction = joblib.load(os.path.join(base_path, 'other.obj'))
        self.task = diction['task']
        self.pad_after = diction['pad_after']
        self.pad_before = diction['pad_before']

    def sample_labels(self, dic_of_labels_limits):
        sampled_targets = self.targets
        sampled_inputs = self.inputs

        for l in dic_of_labels_limits.keys():
            label_set = [i for i in range(len(sampled_targets))
                         if sampled_targets[i][self.task.output_labels.index(l)] == 1]
            if len(label_set) > dic_of_labels_limits[l]:
                random_label_set = random.sample(label_set, dic_of_labels_limits[l])
                sampled_targets = [sampled_targets[i] for i in range(len(sampled_targets)) if
                                   (i not in label_set or i in random_label_set)]
                sampled_inputs = [sampled_inputs[i] for i in range(len(sampled_inputs)) if
                                  (i not in label_set or i in random_label_set)]
        self.inputs = sampled_inputs
        self.targets = sampled_targets

    def k_folds(self, dic_of_labels_limits):
        # Examples
        # --------
        # >> > import numpy as np
        # >> > from sklearn.model_selection import KFold
        # >> > X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        # >> > y = np.array([1, 2, 3, 4])
        # >> > kf = KFold(n_splits=2)
        # >> > kf.get_n_splits(X)
        # 2
        # >> > print(kf)
        # KFold(n_splits=2, random_state=None, shuffle=False)
        # >> > for train_index, test_index in kf.split(X):
        #     ...
        #     print("TRAIN:", train_index, "TEST:", test_index)
        # ...
        # X_train, X_test = X[train_index], X[test_index]
        # ...
        # y_train, y_test = y[train_index], y[test_index]
        # TRAIN: [2 3]
        # TEST: [0 1]
        # TRAIN: [0 1]
        # TEST: [2 3]

        kf = KFold(n_splits=5, shuffle=True)
        if dic_of_labels_limits:
            self.sample_labels(dic_of_labels_limits)
        inputs = self.inputs
        return kf.split(inputs)

    def get_split_by_index(self, train_index, test_index, extraction_method, **kwargs):
        '''

        :param train_index: indexes of the training set
        :param test_index: indexes of the test set
        :param extraction_method: extraction_method object
        :param kwargs: extra arguments for prepare_inputs_targets, namely window_size and window_hop
        :return: the taskdataset for the training and test data
        '''

        x_train = [self.inputs[i] for i in range(len(self.inputs)) if i in train_index]
        y_train = [self.targets[i] for i in range(len(self.targets)) if i in train_index]
        x_val = [self.inputs[i] for i in range(len(self.inputs)) if i in test_index]
        y_val = [self.targets[i] for i in range(len(self.targets)) if i in test_index]

        extraction_method.scale_fit(x_train)
        x_train, y_train = extraction_method.prepare_inputs_targets(x_train, y_train, **kwargs)
        train_taskdataset = TaskDataset(inputs=x_train, targets=y_train, name=self.task.name + "_train",
                                        labels=self.task.output_labels, output_module=self.task.output_module)
        x_val, y_val = extraction_method.prepare_inputs_targets(x_val, y_val, **kwargs)
        test_taskdataset = TaskDataset(inputs=x_val, targets=y_val, name=self.task.name + "_test",
                                       labels=self.task.output_labels, output_module=self.task.output_module)
        return train_taskdataset, test_taskdataset

    @staticmethod
    def check(base_path, extraction_method):
        return os.path.isfile(os.path.join(base_path, '{}_inputs.gz'.format(extraction_method))) and os.path.isfile(
            os.path.join(base_path, 'targets.pt')) and os.path.isfile(os.path.join(base_path, 'other.obj'))
