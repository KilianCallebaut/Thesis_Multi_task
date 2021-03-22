import os

import joblib
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
import tensorflow_datasets as tfds

from Tasks.TaskDataset import TaskDataset


class SpeechCommands(DataReader):
    audioset_path = r"E:\Thesis_Results\Data_Readers\SpeechCommands_{}"
    root = r"E:\Thesis_Datasets\SpeechCommands"

    def __init__(self, extraction_method, test_size=0.2, **kwargs):
        print('start Speech commands')
        self.sample_rate = 16000
        if self.checkfiles(extraction_method):
            self.readfiles(extraction_method)
        else:
            self.loadfiles()
            self.calculateTaskDataset(extraction_method, **kwargs)
            self.writefiles(extraction_method)
        self.split_train_test(test_size=test_size, extraction_method=extraction_method)
        print('Done loading Speech Commands')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'SpeechCommands.obj')

    def get_base_path(self):
        return self.audioset_path.format('train')

    def get_eval_path(self):
        return os.path.join(self.get_base_path(), 'SpeechCommands.obj')

    def get_eval_base_path(self):
        return self.audioset_path.format('eval')

    def checkfiles(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and\
               TaskDataset.check(self.get_eval_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def loadfiles(self):
        self.ds = tfds.load('speech_commands', split=['train', 'test'], shuffle_files=True,
                            data_dir=self.root)
        print('Done loading Speech Commands dataset')

    def readfiles(self, extraction_method):
        self.loadfiles()
        self.taskDataset = TaskDataset([], [], '', [])
        self.taskDataset.load(self.get_base_path(), extraction_method)

        self.validTaskDataset = TaskDataset([], [], '', [])
        self.validTaskDataset.load(self.get_eval_base_path(), extraction_method)

    def writefiles(self, extraction_method):
        dict = {}
        joblib.dump(dict, self.get_path())
        joblib.dump(dict, self.get_eval_path())
        self.taskDataset.save(self.get_base_path(), extraction_method=extraction_method)
        self.validTaskDataset.save(self.get_eval_base_path(), extraction_method=extraction_method)

    def calculate_input(self, method, **kwargs):
        inputs = []
        ds_id = 0

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        if 'test' in kwargs and kwargs.pop('test'):
            ds_id = 1

        for audio_label in self.ds[ds_id].as_numpy_iterator():
            audio = audio_label["audio"]
            fs = self.sample_rate
            if resample_to is not None:
                audio, fs = self.resample(audio, self.sample_rate, resample_to)
            inputs.append(self.extract_features(method, (audio, fs), **kwargs))

        # return self.standardize_input(inputs)
        return inputs

    def calculateTaskDataset(self, method, **kwargs):
        # Training Set
        targets = []
        for audio_label in self.ds[0].as_numpy_iterator():
            targets.append(audio_label["label"])
        distinct_targets = list(set(targets))

        inputs = self.calculate_input(method, **kwargs)
        targets = [[float(b == f) for b in distinct_targets] for f in targets]
        self.taskDataset = TaskDataset(inputs=inputs, targets=targets, name="SpeechCommands", labels=distinct_targets,
                                       output_module='softmax')

        # Test Set
        targets_t = []
        for audio_label in self.ds[1].as_numpy_iterator():
            targets_t.append(audio_label["label"])
        inputs_t = self.calculate_input(method=method, test=True, **kwargs)
        targets_t = [[float(b == f) for b in distinct_targets] for f in targets_t]
        self.validTaskDataset = TaskDataset(inputs=inputs_t, targets=targets_t,
                                            name="SpeechCommandsTest",
                                            labels=distinct_targets, output_module='softmax')

    def recalculate_features(self, method, **kwargs):
        self.taskDataset.inputs = self.calculate_input(method, **kwargs)
        self.validTaskDataset.inputs = self.calculate_input(method, test=True, **kwargs)

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
        self.validTaskDataset.inputs = self.scale_transform(self.validTaskDataset.inputs, extraction_method)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        return self.validTaskDataset
