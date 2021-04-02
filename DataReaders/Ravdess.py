import os

import joblib
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
from Tasks.TaskDataset import TaskDataset


class Ravdess(DataReader):
    object_path = r"E:\Thesis_Results\Data_Readers\Ravdess"
    root = r"E:\Thesis_Datasets\Ravdess"

    def __init__(self, extraction_method, **kwargs):
        self.extraction_method = extraction_method

        print('start ravdess')
        if 'object_path' in kwargs:
                  self.object_path = kwargs.pop('object_path')
        if self.check_files(extraction_method.name):
            self.read_files(extraction_method.name)
        else:
            self.load_files()
            self.calculate_taskDataset(extraction_method, **kwargs)
            self.write_files(extraction_method.name)

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
                             'emotion': em, # (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
                             'emotional_intensity': emi,  # (01 = normal, 02 = strong)
                             'statement': stat, # 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"
                             'repetition': rep,  # 1st or 2nd rep
                             'actor': act,  # Odd numbered are male, even female
                             'file': os.path.join(self.root, song_folder, ss_dir, file)
                             }
                        )

    def read_files(self, extraction_method):
        info = joblib.load(self.get_path())
        self.files = info['files']
        self.taskDataset = TaskDataset([], [], '', [])
        self.taskDataset.load(self.get_base_path(), extraction_method)

    def write_files(self, extraction_method):
        dict = {'files': self.files}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path(), extraction_method)

    def calculate_input(self, method, **kwargs):
        inputs = []
        perc = 0

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for file_idx in range(len(self.files)):
            file = self.files[file_idx]
            read_wav = self.load_wav(file['file'], resample_to)
            inputs.append(method.extract_features(read_wav, **kwargs))
            if perc < (file_idx / len(self.files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        return inputs

    def calculate_taskDataset(self, method, **kwargs):
        print('Calculating input')
        inputs = self.calculate_input(method, **kwargs)

        targets = [f['emotion'] for f in self.files]
        distinct_targets = list(set(targets))
        targets = [[float(b == f) for b in distinct_targets] for f in targets]

        self.taskDataset = TaskDataset(inputs=inputs, targets=targets, name="Ravdess", labels=distinct_targets,
                                       output_module='softmax')

    def recalculate_features(self, method, **kwargs):
        self.taskDataset.inputs = self.calculate_input(method, **kwargs)

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
                                            output_module=self.taskDataset.task.output_module)
        if test_size > 0:
            x_val, y_val = self.extraction_method.prepare_inputs_targets(x_val, y_val, **kwargs)
            self.testTaskDataset = TaskDataset(inputs=x_val, targets=y_val,
                                               name=self.taskDataset.task.name + "_test",
                                               labels=self.taskDataset.task.output_labels,
                                               output_module=self.taskDataset.task.output_module)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        pass
