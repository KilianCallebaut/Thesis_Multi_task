import os

import joblib
import pandas as pd
from numpy import long
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
from DataReaders.ExtractionMethod import extract_options
from Tasks.TaskDataset import TaskDataset


class FSDKaggle2018(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\FSDKaggle2018"
    # object_path = r"E:\Thesis_results\Data_Readers\FSDKaggle2018"
    root = r"E:\Thesis_Datasets\FSDKaggle2018\freesound-audio-tagging"
    audio_folder = r"audio_train"

    def __init__(self, extraction_method, **kwargs):
        print('start FSDKaggle 2018')
        super().__init__(extraction_method, **kwargs)
        print('done FSDKaggle 2018')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'FSDKaggle2018.obj')

    def get_base_path(self):
        return self.object_path

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def load_files(self):
        self.file_labels = pd.read_csv(os.path.join(self.root, 'train.csv'))

    def read_files(self):
        # info = joblib.load(self.get_path())
        # self.file_labels = info['file_labels']
        self.taskDataset.load(self.get_base_path())

    def write_files(self):
        dict = {'file_labels': self.file_labels}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path())

    def calculate_input(self, **kwargs):
        inputs = []
        perc = 0

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for audio_idx in range(len(self.file_labels)):
            path = os.path.join(self.root, self.audio_folder, self.file_labels.loc[audio_idx].fname)
            read_wav = self.load_wav(path, resample_to)
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
        inputs = self.calculate_input(**kwargs)
        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       name='FSDKaggle2018',
                                       labels=distinct_labels,
                                       extraction_method=self.extraction_method,
                                       base_path=self.get_base_path(),
                                       output_module='softmax',
                                       index_mode=self.index_mode)

    def prepare_taskDatasets(self, test_size, dic_of_labels_limits, **kwargs):
        inputs = self.taskDataset.inputs
        targets = self.taskDataset.targets
        if dic_of_labels_limits:
            inputs, targets = self.sample_labels(self.taskDataset, dic_of_labels_limits)

        x_train, x_val, y_train, y_val = \
            train_test_split(inputs, targets, test_size=test_size) \
                if test_size > 0 else (inputs, [], targets, [])
        self.extraction_method.scale_fit(x_train)
        x_train, y_train = self.extraction_method.prepare_inputs_targets(x_train, y_train, **kwargs)
        self.trainTaskDataset = TaskDataset(inputs=x_train, targets=y_train,
                                            name=self.taskDataset.task.name + "_train",
                                            labels=self.taskDataset.task.output_labels,
                                            extraction_method=self.extraction_method,
                                            base_path=self.get_base_path(),
                                            output_module=self.taskDataset.task.output_module,
                                            index_mode=self.index_mode)
        if test_size > 0:
            x_val, y_val = self.extraction_method.prepare_inputs_targets(x_val, y_val, **kwargs)
            self.testTaskDataset = TaskDataset(inputs=x_val, targets=y_val,
                                               name=self.taskDataset.task.name + "_test",
                                               labels=self.taskDataset.task.output_labels,
                                               extraction_method=self.extraction_method,
                                               base_path=self.get_base_path(),
                                               output_module=self.taskDataset.task.output_module,
                                               index_mode=self.index_mode)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        pass
