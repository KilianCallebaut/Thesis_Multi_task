import os

import tensorflow_datasets as tfds

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


class SpeechCommands(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\SpeechCommands_{}"
    # object_path = r"E:\Thesis_Results\Data_Readers\SpeechCommands_{}"
    root = r"F:\Thesis_Datasets\SpeechCommands"

    def __init__(self, extraction_method, **kwargs):
        self.sample_rate = 16000
        print('start Speech commands')
        super().__init__(extraction_method, **kwargs)
        print('Done loading Speech Commands')

    def get_base_path(self):
        return dict(training_base_path=self.object_path.format('train'),
                    testing_base_path=self.object_path.format('eval'))

    def load_files(self):
        self.ds, self.ds_info = tfds.load('speech_commands', split=['train', 'test'], shuffle_files=False,
                                          data_dir=self.root, with_info=True)
        print('Done loading Speech Commands dataset')

    def calculate_input(self, files, resample_to=None):
        inputs = []
        for audio_label in files.as_numpy_iterator():
            audio = audio_label["audio"]
            fs = self.sample_rate
            if resample_to is not None:
                audio, fs = self.resample(audio, self.sample_rate, resample_to)
            inputs.append(self.extraction_method.extract_features((audio, fs)))
            print('input amount: {}'.format(len(inputs)), end='\r')

        return inputs

    def calculate_taskDataset(self, **kwargs) -> HoldTaskDataset:
        # Training Set
        print("Calculate Training Set")
        targets = []
        for audio_label in self.ds[0].as_numpy_iterator():
            targets.append(audio_label["label"])

        inputs = self.calculate_input(self.ds[0], **kwargs)
        inputs_t = self.calculate_input(self.ds[1], **kwargs)
        targets = [[float(b == f) for b in range(len(self.ds_info.features['label'].names))] for f in targets]

        print("Calculate Test Set")
        # Test Set
        targets_t = []
        for audio_label in self.ds[1].as_numpy_iterator():
            targets_t.append(audio_label["label"])
        targets_t = [[float(b == f) for b in range(len(self.ds_info.features['label'].names))] for f in targets_t]

        taskDataset = self.__create_taskDataset__()
        taskDataset.initialize_train_test(
            task=MultiClassTask(name="SpeechCommands",
                                output_labels=self.ds_info.features['label'].names),
            training_inputs=inputs,
            training_targets=targets,
            testing_inputs=inputs_t,
            testing_targets=targets_t
        )
        # taskDataset.prepare_inputs()
        return taskDataset
