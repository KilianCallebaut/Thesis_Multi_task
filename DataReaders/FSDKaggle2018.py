import os
from typing import List

import pandas as pd
import torch
from numpy import long

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


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
        return os.path.join(self.get_base_path()['training_base_path'], 'FSDKaggle2018.obj')

    def get_eval_path(self):
        return os.path.join(self.get_base_path()['testing_base_path'], 'FSDKaggle2018.obj')

    def get_base_path(self):
        return dict(training_base_path=self.object_path.format('train'),
                    testing_base_path=self.object_path.format('eval'))

    def load_files(self):
        self.file_labels = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.file_labels_val = pd.read_csv(os.path.join(self.root, 'test_post_competition.csv'))

    def calculate_input(self, files, resample_to=None) -> List[torch.tensor]:
        inputs = []
        perc = 0

        # if test:
        #     folder_path = os.path.join(self.root, 'audio_test')
        #     files = self.file_labels_val
        # else:
        #     folder_path = os.path.join(self.root, self.audio_folder)
        #     files = self.file_labels

        for audio_idx in range(len(files)):
            path = files[audio_idx]
            read_wav = self.load_wav(path, resample_to)
            if not read_wav:
                inputs.append(torch.tensor([]))
                continue
            inputs.append(self.extraction_method.extract_features(read_wav))
            if perc < (audio_idx / len(self.file_labels)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        return inputs

    def calculate_taskDataset(self, **kwargs) -> HoldTaskDataset:
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

        inputs = self.calculate_input(
            files=[os.path.join(self.root, self.audio_folder, name) for name in self.file_labels['fname']], **kwargs)
        inputs_val = self.calculate_input(
            files=[os.path.join(self.root, 'audio_test', name) for name in self.file_labels_val['fname']], **kwargs)

        for i_id in range(len(inputs)):
            if len(inputs[i_id]) == 0:
                inputs.remove(inputs[i_id])
                targets.remove(targets[i_id])
        for i_id in range(len(inputs_val)):
            if len(inputs_val[i_id]) == 0:
                inputs_val.remove(inputs_val[i_id])
                targets_val.remove(targets_val[i_id])
        taskDataset = self.__create_taskDataset__()
        taskDataset.initialize_train_test(
            task=MultiClassTask(
                name='FSDKaggle2018',
                output_labels=distinct_labels),
            training_inputs=inputs,
            training_targets=targets,
            testing_inputs=inputs_val,
            testing_targets=targets_val
        )
        # taskDataset.prepare_inputs()
        return taskDataset
