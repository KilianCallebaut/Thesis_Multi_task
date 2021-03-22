import os

import joblib
import pandas as pd
from numpy import long
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
from Tasks.TaskDataset import TaskDataset


class FSDKaggle2018(DataReader):
    audioset_path = r"E:\Thesis_results\Data_Readers\FSDKaggle2018"
    root = r"E:\Thesis_Datasets\FSDKaggle2018\freesound-audio-tagging"
    audio_folder = r"audio_train"

    def __init__(self, extraction_method, test_size=0.2, **kwargs):
        print('start FSDKaggle 2018')
        if self.checkfiles(extraction_method):
            self.readfiles(extraction_method)
        else:
            self.loadfiles()
            self.calculateTaskDataset(extraction_method, **kwargs)
            self.writefiles(extraction_method)
        self.split_train_test(test_size=test_size, extraction_method=extraction_method)
        print('done')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'FSDKaggle2018.obj')

    def get_base_path(self):
        return self.audioset_path

    def checkfiles(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def loadfiles(self):
        self.file_labels = pd.read_csv(os.path.join(self.root, 'train.csv'))

    def readfiles(self, extraction_method):
        info = joblib.load(self.get_path())
        self.file_labels = info['file_labels']
        self.taskDataset = TaskDataset([], [], '', [])
        self.taskDataset.load(self.get_base_path(), extraction_method=extraction_method)

    def writefiles(self, extraction_method):
        dict = {'file_labels': self.file_labels}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path(), extraction_method=extraction_method)

    def calculate_input(self, method, **kwargs):
        inputs = []
        perc = 0

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for audio_idx in range(len(self.file_labels)):
            path = os.path.join(self.root, self.audio_folder, self.file_labels.loc[audio_idx].fname)
            read_wav = self.load_wav(path, resample_to)
            inputs.append(self.extract_features(method, read_wav, **kwargs))
            if perc < (audio_idx / len(self.file_labels)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        # return self.standardize_input(inputs)
        return inputs

    def calculateTaskDataset(self, method, **kwargs):
        distinct_labels = self.file_labels.label.unique()
        distinct_labels.sort()
        targets = []
        for f in self.file_labels.label.to_numpy():
            target = [long(distinct_labels[label_id] == f) for label_id in range(len(distinct_labels))]
            targets.append(target)
        inputs = self.calculate_input(method, **kwargs)
        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       name='FSDKaggle2018',
                                       labels=distinct_labels,
                                       output_module='softmax')

    def recalculate_features(self, method, **kwargs):
        self.taskDataset.inputs = self.calculate_input(method, **kwargs)

    def split_train_test(self, test_size, extraction_method):
        x_train, x_val, y_train, y_val = \
            train_test_split(self.taskDataset.inputs, self.taskDataset.targets, test_size=test_size) \
                if test_size > 0 else (self.taskDataset.inputs, [], self.taskDataset.targets, [])
        self.scale_fit(x_train, extraction_method)
        self.trainTaskDataset = TaskDataset(inputs=self.scale_transform(x_train, extraction_method), targets=y_train,
                                            name=self.taskDataset.task.name + "_train",
                                            labels=self.taskDataset.task.output_labels,
                                            output_module=self.taskDataset.task.output_module)
        if test_size > 0:
            self.testTaskDataset = TaskDataset(inputs=self.scale_transform(x_val, extraction_method), targets=y_val,
                                               name=self.taskDataset.task.name + "_test",
                                               labels=self.taskDataset.task.output_labels,
                                               output_module=self.taskDataset.task.output_module)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        pass
