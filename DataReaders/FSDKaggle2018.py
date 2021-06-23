import os

import joblib
import pandas as pd
import torch
from numpy import long

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
from Tasks.TaskDataset import TaskDataset


class FSDKaggle2018(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\FSDKaggle2018_{}"
    # object_path = r"E:\Thesis_results\Data_Readers\FSDKaggle2018"
    root = r"F:\Thesis_Datasets\FSDKaggle2018\freesound-audio-tagging"
    audio_folder = r"audio_train"

    def __init__(self, extraction_method, **kwargs):
        print('start FSDKaggle 2018')
        super().__init__(extraction_method, **kwargs)
        print('done FSDKaggle 2018')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'FSDKaggle2018.obj')

    def get_base_path(self):
        return self.object_path.format('train')

    def get_eval_path(self):
        return os.path.join(self.get_eval_base_path(), 'FSDKaggle2018.obj')

    def get_eval_base_path(self):
        return self.object_path.format('eval')

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and \
               TaskDataset.check(self.get_eval_base_path(), extraction_method) and \
               os.path.isfile(self.get_path()) and \
               os.path.isfile(self.get_eval_path())

    def load_files(self):
        self.file_labels = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.file_labels_val = pd.read_csv(os.path.join(self.root, 'test_post_competition.csv'))

    def read_files(self):
        # info = joblib.load(self.get_path())
        # self.file_labels = info['file_labels']
        self.taskDataset.load(self.get_base_path())
        self.validTaskDataset = TaskDataset(inputs=[], targets=[],
                                            task=MultiClassTask(name='', output_labels=[]),
                                            extraction_method=self.extraction_method,
                                            base_path=self.get_eval_base_path(),
                                            index_mode=self.index_mode)
        self.validTaskDataset.load(self.get_eval_base_path())

    def write_files(self):
        dict = {'file_labels': self.file_labels}
        joblib.dump(dict, self.get_path())
        dict = {'file_labels_val': self.file_labels_val}
        joblib.dump(dict, self.get_eval_path())
        self.taskDataset.save(self.get_base_path())
        self.validTaskDataset.save(self.get_eval_base_path())

    def calculate_input(self, resample_to=None, **kwargs):
        inputs = []
        perc = 0

        if 'test' in kwargs and kwargs.pop('test'):
            folder_path = os.path.join(self.root, 'audio_test')
            files = self.file_labels_val
        else:
            folder_path = os.path.join(self.root, self.audio_folder)
            files = self.file_labels

        for audio_idx in range(len(files)):
            path = os.path.join(folder_path, files.loc[audio_idx].fname)
            read_wav = self.load_wav(path, resample_to)
            if not read_wav:
                inputs.append(torch.tensor([]))
                continue
            inputs.append(self.extraction_method.extract_features(read_wav, **kwargs))
            if perc < (audio_idx / len(self.file_labels)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        return inputs

    def calculate_taskDataset(self, **kwargs):
        distinct_labels = self.file_labels.label.unique()
        distinct_labels.sort()
        targets = []
        for f in self.file_labels.label.to_numpy():
            target = [long(distinct_labels[label_id] == f) for label_id in range(len(distinct_labels))]
            targets.append(target)
        targets_val = []
        for f in self.file_labels_val.label.to_numpy():
            target = [long(distinct_labels[label_id] == f) for label_id in range(len(distinct_labels))]
            targets_val.append(target)
        inputs = self.calculate_input(**kwargs)
        inputs_val = self.calculate_input(**kwargs, test=True)
        for i_id in range(len(inputs)):
            if len(inputs[i_id]) == 0:
                inputs.remove(inputs[i_id])
                targets.remove(targets[i_id])
        for i_id in range(len(inputs_val)):
            if len(inputs_val[i_id]) == 0:
                inputs_val.remove(inputs_val[i_id])
                targets_val.remove(targets_val[i_id])
        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       task=MultiClassTask(
                                           name='FSDKaggle2018_train',
                                           output_labels=distinct_labels),
                                       extraction_method=self.extraction_method,
                                       base_path=self.get_base_path(),
                                       index_mode=self.index_mode)
        self.validTaskDataset = TaskDataset(inputs=inputs_val,
                                            targets=targets_val,
                                            task=MultiClassTask(
                                                name='FSDKaggle2018_test',
                                                output_labels=distinct_labels),
                                            extraction_method=self.extraction_method,
                                            base_path=self.get_eval_base_path(),
                                            index_mode=self.index_mode)
