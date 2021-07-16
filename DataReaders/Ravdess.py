import os

import joblib

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
from Tasks.TaskDataset import TaskDataset
from Tasks.TaskDatasets.HoldTaskDataset import HoldTaskDataset


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
        if os.path.isfile(self.get_path()):
            info = joblib.load(self.get_path())
            self.files = info['files']
            return
        song_folder = 'Audio_Song_actors_01-24'
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
        self.taskDataset.load()

    def write_files(self):
        dict = {'files': self.files}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save()

    def calculate_input(self, resample_to=None):
        inputs = []
        perc = 0

        resample_to = None

        for file_idx in range(len(self.files)):
            file = self.files[file_idx]
            read_wav = self.load_wav(file['file'], resample_to)
            inputs.append(self.extraction_method.extract_features(read_wav))
            if perc < (file_idx / len(self.files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1
        return inputs

    def calculate_taskDataset(self, **kwargs) -> HoldTaskDataset:
        print('Calculating input')
        inputs = self.calculate_input(**kwargs)

        targets = [f['emotion'] for f in self.files]
        distinct_targets = list(set(targets))
        targets = [[float(b == f) for b in distinct_targets] for f in targets]
        taskDataset = HoldTaskDataset(inputs=inputs, targets=targets,
                                           task=MultiClassTask(name="Ravdess", output_labels=distinct_targets),
                                           extraction_method=self.extraction_method,
                                           base_path=self.get_base_path(),
                                           index_mode=self.index_mode,
                                           grouping=[f['actor'] for f in self.files])
        taskDataset.prepare_inputs()
        return  taskDataset
