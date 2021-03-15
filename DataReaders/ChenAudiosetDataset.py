import os
import random
import winsound
from collections import Counter
from datetime import timedelta
from timeit import default_timer as timer

import joblib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torchaudio

try:
    import cPickle
except BaseException:
    import _pickle as cPickle

from DataReaders.DataReader import DataReader
from Tasks.TaskDataset import TaskDataset
from sklearn.model_selection import train_test_split


class ChenAudiosetDataset(DataReader):
    root = r'E:\Thesis_Datasets\audioset_chen\audioset_filtered'
    train_dir = "balanced_train_segments"
    # audioset_path = r"E:\Thesis_Results\Data_Readers\ChenAudiosetDataset.p"
    audioset_path = r"E:\Thesis_Results\Data_Readers\ChenAudiosetDataset"

    # Bools for choosing random selection because overrepresented in dataset
    limit_speech = True
    limit_other = False

    np_objects = []
    files = []
    wav_files = []
    # log_banks = []

    AllLabels = []
    AllFlatenedFielsLabels = []
    AllFlatenedLabels = []
    AllLabelSets = []
    AllFlatenedLabelSets = []
    CountPerLabel = []
    CountFolderPerLabel = []
    CountFolderPerLabelSet = []

    target_list = []

    def __init__(self, extraction_method, test_size=0.2, **kwargs):
        if self.checkfiles(extraction_method):
            self.readfiles(extraction_method)
        else:
            self.loadfiles()
            self.calculateTaskDataset(extraction_method, **kwargs)
            self.writefiles(extraction_method)
        self.split_train_test(test_size=test_size)
        # self.calculateTaskDataset()
        print('done')

    def get_path(self):
        return os.path.join(self.audioset_path, "ChenAudiosetDataset.obj")

    def get_base_path(self):
        return self.audioset_path

    def checkfiles(self, extraction_method):
        return TaskDataset.check(self.get_base_path(), extraction_method) and os.path.isfile(self.get_path())

    def loadfiles(self):
        files = []
        np_objects = []
        wav_files = []
        print(os.path.join(self.root, self.train_dir))

        for _, dirs, _ in os.walk(os.path.join(self.root, self.train_dir)):
            cdt = len(dirs)
            cd = 0
            cn = 0
            start = timer()

            for dir in dirs:
                perc = (cd / cdt) * 100

                cd += 1
                filedir = []
                np_dir = []
                wav_dir = []
                for file in os.listdir(os.path.join(self.root, self.train_dir, dir)):
                    filepath = os.path.join(self.root, self.train_dir, dir, file)
                    if file.endswith('.npy'):
                        np_obj = np.load(filepath, allow_pickle=True)
                        filedir.append(np_obj.item())
                        np_dir.append(np_obj)

                        wav_loc = np_obj.item()['wav_file'].split(r'/')[6]
                        wav_loc = os.path.join(self.root, self.train_dir, dir, wav_loc)
                        # wav_read = librosa.load(os.path.join(self.root, self.train_dir, dir, wav_loc))
                        wav_dir.append(wav_loc)

                files.append(filedir)
                np_objects.append(np_dir)
                wav_files.append(wav_dir)

                if perc > cn * 10:
                    print((cd / cdt) * 100)
                    end = timer()
                    timedel = end - start
                    print('estimated time: {}'.format(timedelta(seconds=timedel * (10 - cn))))
                    start = end
                    cn += 1
        self.files = files
        self.np_objects = np_objects
        self.wav_files = wav_files

    def readfiles(self, extraction_method):
        # info = cPickle.load(open(self.audioset_path, 'rb'))
        info = joblib.load(self.get_path())
        self.files = info['files']
        self.np_objects = info['np_objects']
        self.wav_files = info['wav_files']

        self.taskDataset = TaskDataset([], [], '', [])
        self.taskDataset.load(self.get_base_path(), extraction_method)

    def writefiles(self, extraction_method):
        dict = {'files': self.files,
                'np_objects': self.np_objects,
                'wav_files': self.wav_files}
        joblib.dump(dict, self.get_path())
        self.taskDataset.save(self.get_base_path(), extraction_method)

    def calculateTaskDataset(self, method, **kwargs):
        print('Calculating filterbanks')
        inputs = self.calculate_input(method, **kwargs)
        print('Filterbanks calculated')

        # # list( len: num_of_folders, dat: list( len: num_of_files, dat: list( len: num_of_labels)))
        # targets = [[[l[2] for l in x['labels']] for x in f] for f in
        #            self.files]
        # distinct_targets = list(set([x for l in targets for f in l for x in f]))
        # # Targets are translated as binary strings with 1 for each target
        # # at the index where it is in distinct_targets order
        # # list( len: num_of_folders, dat: list( len: num_of_files, dat: list( len: num_distinct_labels)))
        # targets = [[[float(b in f) for b in distinct_targets] for f in l] for l in targets]
        # # Will transform to list(torch(num_time_steps, num_distinct_labels))

        targets = [[l[2] for l in x['labels']] for f in
                   self.files for x in f]
        distinct_targets = list(set([x for l in targets for x in l]))

        # Targets are translated as binary strings with 1 for each target
        # at the index where it is in distinct_targets order
        targets = [[float(b in f) for b in distinct_targets] for f in targets]

        name = "chen_audioset"
        self.taskDataset = TaskDataset(inputs=inputs, targets=targets, name=name, labels=distinct_targets,
                                       output_module='sigmoid')

    def calculate_input(self, method, **kwargs):
        inputs = []

        resample_to = None
        if 'resample_to' in kwargs:
            resample_to = kwargs.pop('resample_to')

        for folder in self.wav_files:
            for file in folder:
                read_wav = self.load_wav(file, resample_to)
                inputs.append(self.extract_features(method, read_wav, **kwargs))
        # return self.standardize_input(inputs)
        return inputs

    def recalculate_features(self, method, **kwargs):
        self.taskDataset.inputs = self.calculate_input(method, **kwargs)

    def sample_label(self, taskDataset):
        limit = 500
        sampled_targets = taskDataset.targets
        sampled_inputs = taskDataset.inputs

        speech_set = [i for i in range(len(sampled_targets))
                      if sampled_targets[i][taskDataset.task.output_labels.index('Speech')] == 1]
        other_set = [i for i in range(len(sampled_targets))
                     if sampled_targets[i][taskDataset.task.output_labels.index('None of the above')] == 1]

        if self.limit_speech and limit < len(speech_set):
            random_speech_set = random.sample(speech_set, limit)
            non_speech_set = [i for i in range(len(sampled_targets))
                              if sampled_targets[i][taskDataset.task.output_labels.index('Speech')] != 1]
            sampled_targets = [sampled_targets[i] for i in sorted(random_speech_set + non_speech_set)]
            sampled_inputs = [sampled_inputs[i] for i in sorted(random_speech_set + non_speech_set)]

        if self.limit_other and limit < len(other_set):
            random_other_set = random.sample(other_set, limit)
            non_other_set = [i for i in range(len(sampled_targets))
                             if sampled_targets[i][taskDataset.task.output_labels.index('None of the above')] != 1]
            sampled_targets = [sampled_targets[i] for i in sorted(random_other_set + non_other_set)]
            sampled_inputs = [sampled_inputs[i] for i in sorted(random_other_set + non_other_set)]

        return sampled_inputs, sampled_targets

    def split_train_test(self, test_size):
        inputs = self.taskDataset.inputs
        targets = self.taskDataset.targets
        if self.limit_speech or self.limit_other:
            inputs, targets = self.sample_label(self.taskDataset)

        x_train, x_val, y_train, y_val = \
            train_test_split(inputs, targets, test_size=test_size) \
                if test_size > 0 else self.taskDataset.inputs, self.taskDataset.targets, [], []
        means, stds = self.calculate_scalars(inputs)
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

    # def sample_label(self):
    #     limit = 500
    #     sampled_targets = self.taskDataset.targets
    #     sampled_inputs = self.taskDataset.inputs
    #
    #     if self.limit_speech:
    #         speech_set = [i for i in range(len(sampled_targets))
    #                       if self.contains_label(sampled_targets[i], 'Speech')]
    #         random_speech_set = random.sample(speech_set, limit)
    #         non_speech_set = [i for i in range(len(sampled_targets))
    #                           if not self.contains_label(sampled_targets[i], 'Speech')]
    #         sampled_targets = [sampled_targets[i] for i in random_speech_set + non_speech_set]
    #         sampled_inputs = [sampled_inputs[i] for i in random_speech_set + non_speech_set]
    #
    #     if self.limit_other:
    #         other_set = [i for i in range(len(sampled_targets))
    #                      if self.contains_label(sampled_targets[i], 'None of the above')]
    #         random_other_set = random.sample(other_set, limit)
    #         non_other_set = [i for i in range(len(sampled_targets))
    #                          if not self.contains_label(sampled_targets[i], 'None of the above')]
    #         sampled_targets = [sampled_targets[i] for i in random_other_set + non_other_set]
    #         sampled_inputs = [sampled_inputs[i] for i in random_other_set + non_other_set]
    #
    #     return sampled_inputs, sampled_targets

    def contains_label(self, folder, label):
        for f in folder:
            if f[self.taskDataset.task.output_labels.index(label)] == 1:
                return True
        return False

    def open_audio(url, filename):
        # Opening file
        # url = "https://pytorch.org/tutorials/_static/img/steam-train-whistle-daniel_simon-converted-from-mp3.wav"
        # filename = "steam-train-whistle-daniel_simon-converted-from-mp3.wav"

        r = requests.get(url)

        with open(filename, 'wb') as f:
            f.write(r.content)

        waveform, sample_rate = torchaudio.load(filename)
        return waveform, sample_rate

    # Returns list (labels in file) of strings (labels)
    def getLabelsFile(self, folder_ind: int, audio_ind_in_folder: int) -> list:
        return [l[2] for l in self.files[folder_ind][audio_ind_in_folder]['labels'] if l[2] != 'None of the above']

    # Returns list (files in folder)  of list (labels in file) of strings (labels)
    def getLabelsFolder(self, folder_ind: int) -> list:
        return [self.getLabelsFile(folder_ind, f) for f in range(0, len(self.files[folder_ind]))]

    # Returns list (folders in dataset) of list (files in folder)  of list (labels in file) of strings (labels)
    def getAllLabels(self) -> list:
        if not self.AllLabels:
            self.AllLabels = [self.getLabelsFolder(f) for f in range(0, len(self.files))]
        return self.AllLabels

    # Returns list (for each file in db) of lists (for each label in in file) of labels
    def getAllFlatenedFilesLabels(self) -> list:
        if not self.AllFlatenedFielsLabels:
            self.AllFlatenedFielsLabels = [files for folders in self.getAllLabels() for files in folders]
        return self.AllFlatenedFielsLabels

    # Returns list (each label of each file in database) of labels
    def getAllFlatenedLabels(self) -> list:
        if not self.AllFlatenedLabels:
            self.AllFlatenedLabels = [label for files in self.getAllFlatenedFilesLabels() for label in files]
        return self.AllFlatenedLabels

    # Returns list of distinct labels in folder
    def getLabelSet(self, folder_ind: int) -> list:
        return list(set([label for files in self.getLabelsFolder(folder_ind) for label in files]))

    # Returns list of labelsets
    def getAllLabelSets(self) -> list:
        if not self.AllLabelSets:
            self.AllLabelSets = [self.getLabelSet(folder_ind) for folder_ind in range(0, len(self.files))]
        return self.AllLabelSets

    # Returns flatened list of labelsets
    def getAllFlatenedLabelSets(self) -> list:
        if not self.AllFlatenedLabelSets:
            self.AllFlatenedLabelSets = [label for labelset in self.getAllLabelSets() for label in labelset]
        return self.AllFlatenedLabelSets

    # Returns dictionary of occurence counts per label in db
    def getCountPerLabel(self) -> dict:
        if not self.CountPerLabel:
            self.CountPerLabel = dict(sorted(dict(Counter(self.getAllFlatenedLabels())).items(), key=lambda x: x[1]))
        return self.CountPerLabel

    # Returns dictionary with the count of occurence in folders per label
    def getCountFolderPerLabel(self) -> dict:
        if not self.CountFolderPerLabel:
            self.CountFolderPerLabel = dict(
                sorted(dict(Counter(self.getAllFlatenedLabelSets())).items(), key=lambda x: x[1]))
        return self.CountFolderPerLabel

    # Returns dictionary of occurrence in folder counts per labelset
    def getCountFolderPerLabelSet(self) -> dict:
        if not self.CountFolderPerLabelSet:
            self.CountFolderPerLabelSet = dict(
                sorted(dict(Counter(map(tuple, self.getAllLabelSets()))).items(), key=lambda x: x[1]))
        return self.CountFolderPerLabelSet

    # Returns the labels given label occurs along side of
    def findLabelsOccuringWith(self, label) -> list:
        return [y for y in list(set(self.getAllFlatenedLabels())) if
                y in [z for x in self.getCountFolderPerLabelSet() if label in x for z in x]]

    def findLabelsNotOccuringWith(self, label) -> list:
        return [y for y in list(set(self.getAllFlatenedLabels())) if
                y not in [z for x in self.getCountFolderPerLabelSet() if label in x for z in x]]

    def findLabelsOccuringAtSameTimeWith(self, label) -> list:
        return [y for y in list(set(self.getAllFlatenedLabels())) if
                y in [l for folder in self.getAllLabels() for file in folder for labels in file if
                      label in labels and len(labels) > 1 for l in labels]]

    def findLabelsNotOccuringAtSameTimeWith(self, label) -> list:
        return [y for y in list(set(self.getAllFlatenedLabels())) if
                y not in [l for folder in self.getAllLabels() for file in folder for labels in file if
                          label in labels and len(labels) > 1 for l in labels]]

    # def findFilesWithLabels(self, labels: list) ->list:

    def findIndexesFoldersWithLabels(self, labels: list) -> list:
        return [fold_idx for fold_idx in range(0, len(self.files)) if
                len(set(self.getAllLabelSets()[fold_idx]).intersection(labels)) > 0]

    def findFoldersWithLabels(self, labels: list) -> list:
        return [self.files[x] for x in self.findIndexesFoldersWithLabels(labels)]

    def playAudioFiles(self, toplayindexes: list):
        for i in toplayindexes:
            for w in self.files[i]:
                wav = w['wav_file'].replace('/home/cj/', 'C:\\Users\\mrKC1\\Documents\\TU Delft\\Thesis '
                                                         '2\\Dataset\\audioset\\')
                print(wav)
                print(w['labels'])
                winsound.PlaySound(wav, winsound.SND_FILENAME)

    def printStats(self):
        # total count
        print('total folders: {}'.format(len(self.files)))
        print('total files: {}'.format(len(self.getAllFlatenedFilesLabels())))

        # total count per label
        print('counts of labels: {}'.format(self.getCountPerLabel()))
        # histogram of labels, sorted by highest count
        self.showBarChart(self.getCountPerLabel())

        # Count sequences per label
        print('counts of sequences per sets of labels: {}'.format(self.getCountFolderPerLabel()))

        # Count sequences per groups of labels
        # Show label sets and counts grouped per label
        print('sequence count per set of labels grouped by label: {}'.format(self.getCountFolderPerLabelSet()))

        # Show bar chart, grouping by label, stacking the sets
        # 1 bar = (positions = single labels, numbers = counts of set)
        # for each set plot a bar with

    def showHistogram(self, data):
        n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Count of labels')
        pass

    def showBarChart(self, data: dict):
        plt.bar(*zip(*data.items()))
        plt.xticks(rotation=90)
        plt.show()

    def showStackedFor(self):
        bottomdict = dict(zip(list(set(self.flattenedlabels)), [0 for _ in set(self.flattenedlabels)]))

        for l in self.labelsetcounts:
            x = list(l[0])
            heights = [l[1] for _ in x]
            bottoms = [bottomdict[lab] for lab in x]
            for labid in range(0, len(x)):
                bottomdict[x[labid]] += heights[labid]
            plt.bar(x, heights, bottom=bottoms)

        # plt.legend([l[0] for l in self.labelsetcounts], loc=3)
        plt.xticks(rotation=90)
        plt.show()
