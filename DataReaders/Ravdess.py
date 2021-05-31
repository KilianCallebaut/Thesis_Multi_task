import os

import joblib
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
from DataReaders.ExtractionMethod import extract_options
from Tasks.TaskDataset import TaskDataset


class Ravdess(DataReader):
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\Ravdess"
    # object_path = r"E:\Thesis_Results\Data_Readers\Ravdess"
    root = r"F:\Thesis_Datasets\Ravdess"

    def __init__(self, extraction_method, **kwargs):
        print('start ravdess')
        super().__init__(extraction_method, **kwargs)
        print('Done loading Ravdess')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'Ravdess.obj')

    def get_base_path(self):
        return self.object_path

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def load_files(self):
        song_folder = 'Audio_song_actors_01-24'
        speech_folder = 'Audio_Speech_Actors_01-24'
        self.files = []
        for fold in [song_folder, speech_folder]:
            for _, songs, _ in os.walk(os.path.join(self.root, fold)):
                for ss_dir in songs:
                    for file in os.listdir(os.path.join(self.root, song_folder, ss_dir)):
                        mod, voc, em, emi, stat, rep, act = file[:-4].split('-')
                        self.files.append(
                            {'modality': mod,  # (01 = full-AV, 02 = video-only, 03 = audio-only)
                             'vocal_channel': voc,  # (01 = speech, 02 = song)
                             'emotion': em,
                             # (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
                             'emotional_intensity': emi,  # (01 = normal, 02 = strong)
                             'statement': stat,
                             # 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
                             'repetition': rep,  # 1st or 2nd rep
                             'actor': act,  # Odd numbered are male, even female
                             'file': os.path.join(self.root, song_folder, ss_dir, file)
                             }
                        )

    def read_files(self):
        # info = joblib.load(self.get_path())
        # self.files = info['files']
        self.taskDataset.load(self.get_base_path())

    def write_files(self):
        dict = {'files': self.files}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path())

    def calculate_input(self, **kwargs):
        inputs = []
        perc = 0

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for file_idx in range(len(self.files)):
            file = self.files[file_idx]
            read_wav = self.load_wav(file['file'], resample_to)
            inputs.append(self.extraction_method.extract_features(read_wav, **kwargs))
            if perc < (file_idx / len(self.files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        return inputs

    def calculate_taskDataset(self, **kwargs):
        print('Calculating input')
        inputs = self.calculate_input(**kwargs)

        targets = [f['emotion'] for f in self.files]
        distinct_targets = list(set(targets))
        targets = [[float(b == f) for b in distinct_targets] for f in targets]
        self.taskDataset = TaskDataset(inputs=inputs, targets=targets, name="Ravdess", labels=distinct_targets,
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
