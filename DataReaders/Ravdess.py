import os

import joblib
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
from Tasks.TaskDataset import TaskDataset


class Ravdess(DataReader):
    audioset_path = r"E:\Thesis_Results\Data_Readers\Ravdess"
    root = r"E:\Thesis_Datasets\Ravdess"

    def __init__(self, extraction_method, test_size=0.2, **kwargs):
        if self.checkfiles(extraction_method):
            self.readfiles(extraction_method)
        else:
            self.loadfiles()
            self.calculateTaskDataset(extraction_method, **kwargs)
            self.writefiles(extraction_method)
        self.split_train_test(test_size=test_size)
        print('Done loading Ravdess')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'Ravdess.obj')

    def get_base_path(self):
        return self.audioset_path

    def checkfiles(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def loadfiles(self):
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

    def readfiles(self, extraction_method):
        info = joblib.load(self.get_path())
        self.files = info['files']
        self.taskDataset = TaskDataset([], [], '', [])
        self.taskDataset.load(self.get_base_path(), extraction_method)

    def writefiles(self, extraction_method):
        dict = {'files': self.files}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path(), extraction_method)

    def calculate_input(self, method, **kwargs):
        inputs = []

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for file in self.files:
            read_wav = self.load_wav(file['file'], resample_to)
            inputs.append(self.extract_features(method, read_wav, **kwargs))
        # return self.standardize_input(inputs)
        return inputs

    def calculateTaskDataset(self, method, **kwargs):
        print('Calculating input')
        inputs = self.calculate_input(method, **kwargs)

        targets = [f['emotion'] for f in self.files]
        distinct_targets = list(set(targets))
        targets = [[float(b == f) for b in distinct_targets] for f in targets]

        self.taskDataset = TaskDataset(inputs=inputs, targets=targets, name="Ravdess", labels=distinct_targets,
                                       output_module='softmax')

    def recalculate_features(self, method, **kwargs):
        self.taskDataset.inputs = self.calculate_input(method, **kwargs)

    def split_train_test(self, test_size):
        x_train, x_val, y_train, y_val = \
            train_test_split(self.taskDataset.inputs, self.taskDataset.targets, test_size=test_size) \
                if test_size > 0 else self.taskDataset.inputs, self.taskDataset.targets, [], []
        means, stds = self.calculate_scalars(self.taskDataset.inputs)
        self.trainTaskDataset = TaskDataset(inputs=self.standardize_input(x_train, means, stds), targets=y_train,
                                            name=self.taskDataset.task.name + "_train",
                                            labels=self.taskDataset.task.output_labels,
                                            output_module=self.taskDataset.task.output_module)
        self.testTaskDataset = TaskDataset(inputs=self.standardize_input(x_val, means, stds), targets=y_val,
                                           name=self.taskDataset.task.name + "_test",
                                           labels=self.taskDataset.task.output_labels,
                                           output_module=self.taskDataset.task.output_module)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        pass
