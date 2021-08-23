import os

import pandas as pd
from numpy import long

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class FSDKaggle2018(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\FSDKaggle2018_{}"
    # object_path = r"E:\Thesis_results\Data_Readers\FSDKaggle2018"
    root = r"F:\Thesis_Datasets\FSDKaggle2018\freesound-audio-tagging"
    audio_folder = r"audio_train"

    def __init__(self, **kwargs):
        print('start FSDKaggle 2018')
        super().__init__(**kwargs)
        print('done FSDKaggle 2018')

    def get_path(self):
        return os.path.join(self.get_base_path()['training_base_path'], 'FSDKaggle2018.obj')

    def get_eval_path(self):
        return os.path.join(self.get_base_path()['testing_base_path'], 'FSDKaggle2018.obj')

    def get_base_path(self):
        return dict(base_path=self.object_path.format('train'),
                    testing_base_path=self.object_path.format('eval'))

    def load_files(self):
        self.file_labels = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.file_labels_val = pd.read_csv(os.path.join(self.root, 'test_post_competition.csv'))

    def calculate_input(self, taskDataset: HoldTaskDataset, preprocess_parameters: dict):
        perc = 0
        files = [os.path.join(self.root, self.audio_folder, name) for name in self.file_labels['fname']]

        for audio_idx in range(len(files)):
            path = files[audio_idx]
            read_wav = self.preprocess_signal(self.load_wav(path), **preprocess_parameters)
            taskDataset.add_input(read_wav)

            if perc < (audio_idx / len(self.file_labels)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1

        files = [os.path.join(self.root, 'audio_test', name) for name in self.file_labels_val['fname']]
        for audio_idx in range(len(files)):
            path = files[audio_idx]
            read_wav = self.preprocess_signal(self.load_wav(path), **preprocess_parameters)
            taskDataset.test_set.add_input(read_wav)

            if perc < (audio_idx / len(self.file_labels)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1

    def calculate_taskDataset(self,
                              taskDataset: HoldTaskDataset,
                              **kwargs):
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

        task = MultiClassTask(
            name='FSDKaggle2018',
            output_labels=distinct_labels)
        taskDataset.add_task_and_targets(task=task, targets=targets)
        taskDataset.test_set.add_task_and_targets(task=task, targets=targets_val)
