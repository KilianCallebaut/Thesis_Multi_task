import os

import joblib
import numpy as np
import pandas as pd
from numpy import long

from DataReaders.DataReader import DataReader
from Tasks.Task import MultiClassTask
from Tasks.TaskDataset import TaskDataset

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


class ASVspoof2015(DataReader):
    Wav_folder = r"F:\Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015\DS_10283_853\wav"
    Label_folder = r"F:\Thesis_Datasets\Automatic Speaker Verification Spoofing and Countermeasures Challenge 2015\DS_10283_853"
    object_path = r"C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\ASVspoof2015_{}"

    def __init__(self, extraction_method, **kwargs):
        print('start asvspoof2015')
        super().__init__(extraction_method, **kwargs)
        print('done asvspoof2015')

    def get_path(self):
        return os.path.join(self.get_base_path(), 'ASVspoof2015.obj')

    def get_base_path(self):
        return self.object_path.format('train')

    def check_files(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def load_files(self):
        if os.path.isfile(self.get_path()):
            info = joblib.load(self.get_path())
            self.files = info['files']
            self.truths = info['truths']
            return

        # self.truths = pd.read_csv(os.path.join(self.Label_folder, 'cm_develop.ndx'), sep=' ', header=None,
        #                           names=['folder', 'file', 'method', 'source'])
        # add = pd.read_csv(os.path.join(self.Label_folder, 'cm_evaluation.ndx'), sep=' ', header=None,
        #                   names=['folder', 'file', 'method', 'source'])
        # self.truths = self.truths.append(add)
        self.truths = pd.read_csv(os.path.join(self.Label_folder, 'CM_protocol', 'cm_train.trn'), sep=' ', header=None,
                                  names=['folder', 'file', 'method', 'source'])
        # self.truths = self.truths.append(add)
        truths_male = pd.read_csv(os.path.join(self.Label_folder, 'Joint_ASV_CM_protocol', 'ASV_male_development.ndx'),
                                  sep=' ',
                                  header=None,
                                  names=['folder', 'file', 'method', 'source'])
        truths_female = pd.read_csv(
            os.path.join(self.Label_folder, 'Joint_ASV_CM_protocol', 'ASV_female_development.ndx'), sep=' ',
            header=None,
            names=['folder', 'file', 'method', 'source'])
        male_eval = pd.read_csv(os.path.join(self.Label_folder, 'Joint_ASV_CM_protocol', 'ASV_male_evaluation.ndx'),
                                sep=' ', header=None,
                                names=['folder', 'file', 'method', 'source'])
        female_eval = pd.read_csv(os.path.join(self.Label_folder, 'Joint_ASV_CM_protocol', 'ASV_female_evaluation.ndx'),
                                  sep=' ',
                                  header=None,
                                  names=['folder', 'file', 'method', 'source'])
        male_enrol = pd.read_csv(os.path.join(self.Label_folder, 'Joint_ASV_CM_protocol', 'ASV_male_enrolment.ndx'),
                                 sep=' ', header=None,
                                 names=['folder', 'file', 'method', 'source'])
        female_enrol = pd.read_csv(os.path.join(self.Label_folder, 'Joint_ASV_CM_protocol', 'ASV_female_enrolment.ndx'),
                                   sep=' ',
                                   header=None,
                                   names=['folder', 'file', 'method', 'source'])
        self.truths.append(truths_male)
        self.truths.append(truths_female)
        self.truths.append(male_eval)
        self.truths.append(female_eval)
        self.truths.append(male_enrol)
        self.truths.append(female_enrol)

        # self.truths = self.truths[(self.truths.method == 'human')]
        self.truths.sort_values(['folder', 'file'], inplace=True)

        self.files = [os.path.join(self.Wav_folder, x[0], x[1]) for x in self.truths.to_numpy()]

    def read_files(self):
        # info = joblib.load(self.get_path())
        # self.files = info['files']
        # self.truths = info['truths']
        # self.files_val = info['files_val']
        # self.truths_val = info['truths_val']

        self.taskDataset.load()

    def write_files(self):
        dict = {'files': self.files,
                'truths': self.truths,
                }
        joblib.dump(dict, self.get_path())
        self.taskDataset.save()

    # which = develop, evaluation
    def calculate_input(self, resample_to=None, **kwargs):
        print('training')
        perc = 0
        inputs = []
        for audio_idx in range(len(self.files)):
            read_wav = self.load_wav(self.files[audio_idx] + '.wav', resample_to)
            inputs.append(self.extraction_method.extract_features(read_wav, **kwargs))
            if perc < (audio_idx / len(self.files)) * 100:
                print("Percentage done: {}".format(perc))
                perc += 1

        return inputs

    def calculate_taskDataset(self, **kwargs):
        distinct_labels = self.truths.folder.unique()
        distinct_labels.sort()
        distinct_labels = np.append(distinct_labels, 'unknown')

        targets = []
        inputs = self.calculate_input(**kwargs)
        for i in range(len(inputs)):
            target = [long(distinct_labels[label_id] == self.truths.loc[i].folder) if (
                    self.truths.loc[i].method == 'genuine' or self.truths.loc[i].method == 'human') else long(
                label_id == len(distinct_labels) - 1) for label_id in range(len(distinct_labels))]
            targets.append(target)

        self.taskDataset = TaskDataset(inputs=inputs,
                                       targets=targets,
                                       task=MultiClassTask(name='ASVspoof2015',
                                                           output_labels=distinct_labels),
                                       extraction_method=self.extraction_method,
                                       base_path=self.get_base_path(),
                                       index_mode=self.index_mode)
