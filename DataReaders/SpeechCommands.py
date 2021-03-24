import os
import random

import joblib
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
import tensorflow_datasets as tfds

from Tasks.TaskDataset import TaskDataset


class SpeechCommands(DataReader):
    object_path = r"E:\Thesis_Results\Data_Readers\SpeechCommands_{}"
    root = r"E:\Thesis_Datasets\SpeechCommands"

    def __init__(self, extraction_method, test_size=0.2, **kwargs):
        print('start Speech commands')

        self.sample_rate = 16000
        if 'object_path' in kwargs:
            self.object_path = kwargs.pop('object_path')
        if self.checkfiles(extraction_method.name):
            self.readfiles(extraction_method.name)
        else:
            self.loadfiles()
            self.calculateTaskDataset(extraction_method, **kwargs)
            self.writefiles(extraction_method.name)
        self.prepare_taskDatasets(test_size=test_size, extraction_method=extraction_method)
        print('Done loading Speech Commands')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'SpeechCommands.obj')

    def get_base_path(self):
        return self.object_path.format('train')

    def get_eval_path(self):
        return os.path.join(self.get_base_path(), 'SpeechCommands.obj')

    def get_eval_base_path(self):
        return self.object_path.format('eval')

    def checkfiles(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and \
               TaskDataset.check(self.get_eval_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def loadfiles(self):
        self.ds, self.ds_info = tfds.load('speech_commands', split=['train', 'test'], shuffle_files=True,
                                          data_dir=self.root, with_info=True)
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
            inputs.append(method.extract_features((audio, fs), **kwargs))

        # inputs = self.pad(inputs, self.max_length(inputs))
        return inputs

    def calculateTaskDataset(self, method, **kwargs):
        # Training Set
        targets = []
        for audio_label in self.ds[0].as_numpy_iterator():
            targets.append(audio_label["label"])
        distinct_targets = list(set(targets))

        inputs = self.calculate_input(method, **kwargs)
        targets = [[float(b == f) for b in distinct_targets] for f in targets]

        self.taskDataset = TaskDataset(inputs=inputs, targets=targets, name="SpeechCommands",
                                       labels=self.ds_info.features['label'].names,
                                       output_module='softmax')

        # Test Set
        targets_t = []
        for audio_label in self.ds[1].as_numpy_iterator():
            targets_t.append(audio_label["label"])
        inputs_t = self.calculate_input(method=method, test=True, **kwargs)
        targets_t = [[float(b == f) for b in distinct_targets] for f in targets_t]

        self.validTaskDataset = TaskDataset(inputs=inputs_t, targets=targets_t,
                                            name="SpeechCommandsTest",
                                            labels=self.ds_info.features['label'].names, output_module='softmax')

    def recalculate_features(self, method, **kwargs):
        self.taskDataset.inputs = self.calculate_input(method, **kwargs)
        self.validTaskDataset.inputs = self.calculate_input(method, test=True, **kwargs)

    def sample_label(self, taskDataset):
        limit = 5000
        sampled_targets = taskDataset.targets
        sampled_inputs = taskDataset.inputs

        other_set = [i for i in range(len(sampled_targets))
                     if sampled_targets[i][-1] == 1]
        non_other_set = [i for i in range(len(sampled_targets))
                         if sampled_targets[i][-1] != 1]
        random_other_set = random.sample(other_set, limit)
        sampled_targets = [sampled_targets[i] for i in sorted(random_other_set + non_other_set)]
        sampled_inputs = [sampled_inputs[i] for i in sorted(random_other_set + non_other_set)]
        return sampled_inputs, sampled_targets

    def prepare_taskDatasets(self, test_size, extraction_method):
        inputs, targets = self.sample_label(self.taskDataset)

        x_train, x_val, y_train, y_val = \
            train_test_split(inputs, targets, test_size=test_size) \
                if test_size > 0 else (inputs, [], targets, [])
        extraction_method.scale_fit(x_train)
        x_train, y_train = extraction_method.prepare_inputs_targets(x_train, y_train)
        self.trainTaskDataset = TaskDataset(inputs=x_train, targets=y_train,
                                            name=self.taskDataset.task.name + "_train",
                                            labels=self.taskDataset.task.output_labels,
                                            output_module=self.taskDataset.task.output_module)
        if test_size > 0:
            x_val, y_val = extraction_method.prepare_inputs_targets(x_val, y_val)
            self.testTaskDataset = TaskDataset(inputs=x_val, targets=y_val,
                                               name=self.taskDataset.task.name + "_test",
                                               labels=self.taskDataset.task.output_labels,
                                               output_module=self.taskDataset.task.output_module)

        self.validTaskDataset.inputs, self.validTaskDataset.targets = \
            extraction_method.prepare_inputs_targets(self.validTaskDataset.inputs, self.validTaskDataset.targets)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        return self.validTaskDataset
