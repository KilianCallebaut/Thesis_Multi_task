import os

import joblib
import tensorflow_datasets as tfds

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
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
        return os.path.join(self.get_eval_base_path(), 'SpeechCommands.obj')

    def get_eval_base_path(self):
        return self.object_path.format('eval')

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and \
               TaskDataset.check(self.get_eval_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def load_files(self):
        self.ds, self.ds_info = tfds.load('speech_commands', split=['train', 'test'], shuffle_files=False,
                                          data_dir=self.root, with_info=True)
        print('Done loading Speech Commands dataset')

    def read_files(self):
        # self.load_files()
        self.taskDataset.load(self.get_base_path())

        self.validTaskDataset = TaskDataset(inputs=[], targets=[],
                                            task=MultiClassTask(name='', output_labels=[]),
                                            extraction_method=self.extraction_method,
                                            base_path=self.get_eval_base_path(),
                                            index_mode=self.index_mode)
        self.validTaskDataset.load(self.get_eval_base_path())

    def write_files(self):
        dict = {}
        joblib.dump(dict, self.get_path())
        joblib.dump(dict, self.get_eval_path())
        self.taskDataset.save(self.get_base_path())
        self.validTaskDataset.save(self.get_eval_base_path())

    def calculate_input(self, **kwargs):
        inputs_tot = [[], []]
        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for ds_id in range(2):
            for audio_label in self.ds[ds_id].as_numpy_iterator():
                audio = audio_label["audio"]
                fs = self.sample_rate
                if resample_to is not None:
                    audio, fs = self.resample(audio, self.sample_rate, resample_to)
                inputs_tot[ds_id].append(self.extraction_method.extract_features((audio, fs), **kwargs))
                print('input amount: {}'.format(len(inputs_tot[ds_id])), end='\r')

        return inputs_tot[0], inputs_tot[1]

    def calculate_taskDataset(self, **kwargs):
        # Training Set
        print("Calculate Training Set")
        targets = []
        for audio_label in self.ds[0].as_numpy_iterator():
            targets.append(audio_label["label"])

        inputs, inputs_t = self.calculate_input(**kwargs)
        targets = [[float(b == f) for b in range(len(self.ds_info.features['label'].names))] for f in targets]

        print("Calculate Test Set")
        # Test Set
        targets_t = []
        for audio_label in self.ds[1].as_numpy_iterator():
            targets_t.append(audio_label["label"])
        targets_t = [[float(b == f) for b in range(len(self.ds_info.features['label'].names))] for f in targets_t]

        self.taskDataset = TaskDataset(inputs=inputs, targets=targets,
                                       task=MultiClassTask(name="SpeechCommands_train",
                                                           output_labels=self.ds_info.features['label'].names),
                                       extraction_method=self.extraction_method, base_path=self.get_base_path(),
                                       index_mode=self.index_mode)
        self.validTaskDataset = TaskDataset(inputs=inputs_t, targets=targets_t,
                                            task=MultiClassTask(name="SpeechCommands_eval",
                                                                output_labels=self.ds_info.features['label'].names),
                                            extraction_method=self.extraction_method,
                                            base_path=self.get_base_path(),
                                            index_mode=self.index_mode)
