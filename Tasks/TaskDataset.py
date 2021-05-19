import os
import pickle
import random
import types

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
            extraction_method,
            base_path,
            output_module='softmax',  # 'sigmoid'
            index_mode=False
    ):
        if index_mode:
            self.switch_index_methods()

        self.inputs = inputs  # List of tensors
        self.targets = targets  # List of binary strings
        # self.name = name # String
        # self.labels = labels # List of strings
        self.task = Task(name, labels, output_module)
        self.extraction_method = extraction_method
        self.base_path = base_path
        self.pad_after = list()
        self.pad_before = list()
        self.index_mode = index_mode

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        return self.inputs[index].float(), \
               torch.from_numpy(np.array(self.pad_before + self.targets[index] + self.pad_after)), \
               self.task.name

    def __len__(self):
        return len(self.inputs)

    def pad_targets(self, before: int, after: int):
        self.pad_before = [0 for _ in range(before)]
        self.pad_after = [0 for _ in range(after)]

    def save(self, base_path):
        joblib.dump(self.inputs, os.path.join(base_path, '{}_inputs.gz'.format(self.extraction_method.name)))
        t_s = torch.Tensor(self.targets)
        torch.save(t_s, os.path.join(base_path, 'targets.pt'))

        diction = {
            'task': self.task,
            'pad_after': self.pad_after,
            'pad_before': self.pad_before
        }
        joblib.dump(diction, os.path.join(base_path, 'other.obj'))

    def load(self, base_path):
        self.inputs = joblib.load(os.path.join(base_path, '{}_inputs.gz'.format(self.extraction_method.name)))
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

    def k_folds(self, dic_of_labels_limits, random_state=None):
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

        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        if dic_of_labels_limits:
            self.sample_labels(dic_of_labels_limits)
        inputs = self.inputs
        return kf.split(inputs)

    def get_split_by_index(self, train_index, test_index, **kwargs):
        '''

        :param train_index: indexes of the training set
        :param test_index: indexes of the test set
        :param kwargs: extra arguments for prepare_inputs_targets, namely window_size and window_hop
        :return: the taskdataset for the training and test data
        '''

        x_train = [self.inputs[i] for i in range(len(self.inputs)) if i in train_index]
        y_train = [self.targets[i] for i in range(len(self.targets)) if i in train_index]
        x_val = [self.inputs[i] for i in range(len(self.inputs)) if i in test_index]
        y_val = [self.targets[i] for i in range(len(self.targets)) if i in test_index]

        if 'fold' and 'random_state' in kwargs:
            fold = kwargs.pop('fold')
            random_state = kwargs.pop('random_state')
            self.load_split_scalers(fold, random_state)
        else:
            self.extraction_method.scale_fit(x_train)

        x_train, y_train = self.extraction_method.prepare_inputs_targets(x_train, y_train, **kwargs)
        train_taskdataset = TaskDataset(inputs=x_train, targets=y_train, name=self.task.name + "_train",
                                        labels=self.task.output_labels, extraction_method=self.extraction_method,
                                        base_path=self.base_path, output_module=self.task.output_module)
        x_val, y_val = self.extraction_method.prepare_inputs_targets(x_val, y_val, **kwargs)
        test_taskdataset = TaskDataset(inputs=x_val, targets=y_val, name=self.task.name + "_test",
                                       labels=self.task.output_labels, extraction_method=self.extraction_method,
                                       base_path=self.base_path, output_module=self.task.output_module)
        return train_taskdataset, test_taskdataset

    # normal_mode method
    def save_split_scalers(self, dic_of_labels_limits, random_state):
        i = 0
        for train, _ in self.k_folds(dic_of_labels_limits, random_state):
            x_train = [self.inputs[i] for i in range(len(self.inputs)) if i in train]
            self.extraction_method.scale_fit(x_train)
            scalers = self.extraction_method.scalers
            path = os.path.join(self.base_path,
                                'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
                                                                                  random_state, i))
            joblib.dump(value=scalers, filename=path, protocol=pickle.HIGHEST_PROTOCOL)
            i += 1

    def load_split_scalers(self, fold, random_state):
        path = os.path.join(self.base_path,
                            'scaler_method_{}_state_{}_fold_{}.pickle'.format(self.extraction_method.name,
                                                                              random_state, fold))
        self.extraction_method.scalers = joblib.load(path)

    def to_index_mode(self, **kwargs):
        # save inputs in separate files (if not done so already)
        self.index_mode = True
        separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
        if not os.path.exists(separated_dir):
            os.mkdir(separated_dir)

        separated_dir_tar = os.path.join(self.base_path, 'target_{}_separated'.format(self.extraction_method.name))
        if not os.path.exists(separated_dir_tar):
            os.mkdir(separated_dir_tar)

        if not os.listdir(separated_dir):
            for i in range(len(self.inputs)):
                windowed_i, windowed_t = self.extraction_method.prepare_inputs_targets([self.inputs[i]],
                                                                                       [self.targets[i]],
                                                                                       **kwargs)
                for w in range(len(windowed_i)):
                    input_path = os.path.join(separated_dir, 'input_{}_window_{}.pickle'.format(i, w))
                    torch.save(windowed_i[w], input_path)
                    target_path = os.path.join(separated_dir_tar, 'target_{}_window_{}.pickle'.format(i, w))
                    joblib.dump(windowed_t[w], target_path, protocol=pickle.HIGHEST_PROTOCOL)

        # replace inputs with index lists
        self.inputs = [i for i in range(len(self.inputs))]
        self.switch_index_methods()

    def switch_index_methods(self):
        # replace getitem, get_split_by_index by index based functions
        self.get_item = types.MethodType(get_item_index_mode, self)
        self.get_split_by_index = types.MethodType(get_split_by_index_index_mode, self)
        self.save = types.MethodType(save_index_mode, self)
        self.load = types.MethodType(load_index_mode, self)

    @staticmethod
    def check(base_path, extraction_method):
        return os.path.isfile(os.path.join(base_path, '{}_inputs.gz'.format(extraction_method))) and os.path.isfile(
            os.path.join(base_path, 'targets.pt')) and os.path.isfile(os.path.join(base_path, 'other.obj'))


def get_item_index_mode(self, index):
    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    x = torch.load(os.path.join(separated_dir, os.listdir(separated_dir)[self.inputs[index]])).float()
    # separated_dir_tar = os.path.join(self.base_path, 'target_{}_separated'.format(self.extraction_method.name))
    # y = joblib.load(os.path.join(separated_dir_tar, os.listdir(separated_dir_tar)[index]))
    y = self.targets[index]
    return x, \
           torch.from_numpy(np.array(self.pad_before + y + self.pad_after)), \
           self.task.name


def get_split_by_index_index_mode(self, train_index, test_index, **kwargs):
    '''
        :param train_index: indexes of the training set
        :param test_index: indexes of the test set
        :param kwargs: extra arguments for prepare_inputs_targets, namely window_size and window_hop
        :return: the taskdataset for the training and test data
    '''

    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    total_inputs = os.listdir(separated_dir)
    x_train_window = [ind for ind in range(len(total_inputs)) if int(total_inputs[ind].split('_')[1]) in train_index]
    x_val_window = [ind for ind in range(len(total_inputs)) if int(total_inputs[ind].split('_')[1]) in test_index]

    separated_dir_tar = os.path.join(self.base_path, 'target_{}_separated'.format(self.extraction_method.name))
    total_targets = os.listdir(separated_dir_tar)
    y_train_window = [joblib.load(os.path.join(separated_dir_tar, total_targets[ind])) for ind in range(len(total_targets)) if
                      int(total_targets[ind].split('_')[1]) in train_index]
    y_val_window = [joblib.load(os.path.join(separated_dir_tar, total_targets[ind])) for ind in range(len(total_targets)) if
                    int(total_targets[ind].split('_')[1]) in test_index]

    if 'fold' and 'random_state' in kwargs:
        fold = kwargs.pop('fold')
        random_state = kwargs.pop('random_state')
        self.load_split_scalers(fold, random_state)
    else:
        x_train = [torch.load(os.path.join(separated_dir, total_inputs[ind])).float()
                   for ind in x_train_window]
        self.extraction_method.scale_fit(x_train)

    train_taskdataset = TaskDataset(inputs=x_train_window, targets=y_train_window, name=self.task.name + "_train",
                                    labels=self.task.output_labels, extraction_method=self.extraction_method,
                                    base_path=self.base_path, output_module=self.task.output_module, index_mode=True)
    test_taskdataset = TaskDataset(inputs=x_val_window, targets=y_val_window, name=self.task.name + "_test",
                                   labels=self.task.output_labels, extraction_method=self.extraction_method,
                                   base_path=self.base_path, output_module=self.task.output_module, index_mode=True)

    return train_taskdataset, test_taskdataset


def save_index_mode(self, base_path):
    separated_dir = os.path.join(self.base_path, '{}_separated'.format(self.extraction_method.name))
    if not os.path.exists(separated_dir):
        os.mkdir(separated_dir)

    separated_dir_tar = os.path.join(self.base_path, 'target_{}_separated'.format(self.extraction_method.name))
    if not os.path.exists(separated_dir_tar):
        os.mkdir(separated_dir_tar)

    if not os.listdir(separated_dir):
        for i in range(len(self.inputs)):
            windowed_i, windowed_t = self.extraction_method.prepare_inputs_targets([self.inputs[i]], [self.targets[i]])
            for w in range(len(windowed_i)):
                input_path = os.path.join(separated_dir, 'input_{}_window_{}.pickle'.format(i, w))
                torch.save(windowed_i, input_path)
                target_path = os.path.join(separated_dir_tar, 'target_{}_window_{}.pickle'.format(i, w))
                joblib.dump(windowed_t, target_path, protocol=pickle.HIGHEST_PROTOCOL)
    self.inputs = [i for i in range(len(self.inputs))]

    diction = {
        'task': self.task,
        'pad_after': self.pad_after,
        'pad_before': self.pad_before
    }
    joblib.dump(diction, os.path.join(base_path, 'other.obj'))


def load_index_mode(self, base_path):
    separated_dir = os.path.join(self.base_path, 'input_{}_separated'.format(self.extraction_method.name))
    dir_list = os.listdir(separated_dir)
    max_frag = 0
    for d in dir_list:
        num = int(d.split('_')[1])
        if num > max_frag:
            max_frag = num

    self.inputs = [i for i in range(max_frag)]
    t_l = torch.load(os.path.join(base_path, 'targets.pt'))
    self.targets = [[int(j) for j in i] for i in t_l]
    diction = joblib.load(os.path.join(base_path, 'other.obj'))
    self.task = diction['task']
    self.pad_after = diction['pad_after']
    self.pad_before = diction['pad_before']
