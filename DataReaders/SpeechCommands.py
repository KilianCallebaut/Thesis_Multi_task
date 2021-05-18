import os

import joblib
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
from DataReaders.ExtractionMethod import extract_options
from Tasks.TaskDataset import TaskDataset


class SpeechCommands(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\SpeechCommands_{}"
    # object_path = r"E:\Thesis_Results\Data_Readers\SpeechCommands_{}"
    root = r"F:\Thesis_Datasets\SpeechCommands"

    def __init__(self, extraction_method, **kwargs):
        self.sample_rate = 16000
        print('start Speech commands')
        super().__init__(extraction_method, **kwargs)
        print('Done loading Speech Commands')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'SpeechCommands.obj')

    def get_base_path(self):
        return self.object_path.format('train')

    def get_eval_path(self):
        return os.path.join(self.get_base_path(), 'SpeechCommands.obj')

    def get_eval_base_path(self):
        return self.object_path.format('eval')

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and \
               TaskDataset.check(self.get_eval_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def load_files(self):
        self.ds, self.ds_info = tfds.load('speech_commands', split=['train', 'test'], shuffle_files=True,
                                          data_dir=self.root, with_info=True)
        print('Done loading Speech Commands dataset')

    def read_files(self):
        # self.load_files()
        self.taskDataset = TaskDataset([], [], '', [], self.extraction_method, base_path=self.get_base_path(),
                                       index_mode=self.index_mode)
        self.taskDataset.load(self.get_base_path())

        # self.validTaskDataset = TaskDataset([], [], '', [])
        # self.validTaskDataset.load(self.get_eval_base_path(), extraction_method)

    def write_files(self):
        dict = {}
        joblib.dump(dict, self.get_path())
        joblib.dump(dict, self.get_eval_path())
        self.taskDataset.save(self.get_base_path())
        self.validTaskDataset.save(self.get_eval_base_path())

    def calculate_input(self, **kwargs):
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
            inputs.append(self.extraction_method.extract_features((audio, fs), **kwargs))

        return inputs

    def calculate_taskDataset(self, **kwargs):
        # Training Set
        print("Calculate Training Set")
        targets = []
        for audio_label in self.ds[0].as_numpy_iterator():
            targets.append(audio_label["label"])
        distinct_targets = list(set(targets))

        inputs = self.calculate_input(**kwargs)
        targets = [[float(b == f) for b in distinct_targets] for f in targets]

        self.taskDataset = TaskDataset(inputs=inputs, targets=targets, name="SpeechCommands",
                                       labels=self.ds_info.features['label'].names,
                                       extraction_method=self.extraction_method, base_path=self.get_base_path(),
                                       output_module='softmax',
                                       index_mode=self.index_mode)

        print("Calculate Test Set")
        # Test Set
        targets_t = []
        for audio_label in self.ds[1].as_numpy_iterator():
            targets_t.append(audio_label["label"])
        inputs_t = self.calculate_input(test=True, **kwargs)
        targets_t = [[float(b == f) for b in distinct_targets] for f in targets_t]

        self.validTaskDataset = TaskDataset(inputs=inputs_t, targets=targets_t,
                                            name="SpeechCommandsTest",
                                            extraction_method=self.extraction_method,
                                            labels=self.ds_info.features['label'].names,
                                            base_path=self.get_base_path(), output_module='softmax',
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

        self.validTaskDataset.inputs, self.validTaskDataset.targets = \
            self.extraction_method.prepare_inputs_targets(self.validTaskDataset.inputs, self.validTaskDataset.targets,
                                                          **kwargs)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        return self.validTaskDataset
