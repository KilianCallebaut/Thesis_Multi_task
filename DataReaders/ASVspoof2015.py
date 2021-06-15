import os

import joblib
import numpy as np
import pandas as pd
from numpy import long
from sklearn.model_selection import train_test_split

from DataReaders.DataReader import DataReader
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

        self.taskDataset.load(self.get_base_path())

    def write_files(self):
        dict = {'files': self.files,
                'truths': self.truths,
                }
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path())

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
                                       name='ASVspoof2015',
                                       labels=distinct_labels,
                                       extraction_method=self.extraction_method,
                                       output_module='softmax',
                                       base_path=self.get_base_path(),
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
                                            output_module=self.taskDataset.task.output_module,
                                            base_path=self.get_base_path(),
                                            index_mode=self.index_mode)
        if test_size > 0:
            x_val, y_val = self.extraction_method.prepare_inputs_targets(x_val, y_val, **kwargs)
            self.testTaskDataset = TaskDataset(inputs=x_val, targets=y_val,
                                               name=self.taskDataset.task.name + "_test",
                                               labels=self.taskDataset.task.output_labels,
                                               extraction_method=self.extraction_method,
                                               output_module=self.taskDataset.task.output_module,
                                               base_path=self.get_base_path(),
                                               index_mode=self.index_mode)

        self.valTaskDataset.inputs, self.valTaskDataset.targets = self.extraction_method.prepare_inputs_targets(
            self.valTaskDataset.inputs, self.valTaskDataset.targets, **kwargs)

    def toTrainTaskDataset(self):
        return self.trainTaskDataset

    def toTestTaskDataset(self):
        return self.testTaskDataset

    def toValidTaskDataset(self):
        pass
