from __future__ import print_function, division

import sys
from datetime import datetime

import torch

from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.DCASE2017_SE import DCASE2017_SE
from DataReaders.DCASE2017_SS import DCASE2017_SS
from DataReaders.FSDKaggle2018 import FSDKaggle2018
from DataReaders.Ravdess import Ravdess
from DataReaders.SpeechCommands import SpeechCommands
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tests.config_reader import *
from Training.Results import Results
from Training.Training import Training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'F'


def run_datasets(dataset_list, extraction_params):
    calculate_window_size(extraction_params)
    taskDatasets = []

    if 0 in dataset_list:
        asvspoof = ASVspoof2015(**extraction_params,
                                object_path=r'F:\Thesis_Results\Data_Readers\ASVspoof2015_{}')
        taskDatasets.append(asvspoof.taskDataset)
    if 1 in dataset_list:
        chenaudio = ChenAudiosetDataset(**extraction_params,
                                        object_path=r'F:\Thesis_Results\Data_Readers\ChenAudiosetDataset')
        taskDatasets.append(chenaudio.taskDataset)
    if 2 in dataset_list:
        dcaseScene = DCASE2017_SS(**extraction_params,
                                  object_path=r'F:\Thesis_Results\Data_Readers\DCASE2017_SS_{}')
        taskDatasets.append(dcaseScene.taskDataset)
    if 3 in dataset_list:
        fsdkaggle = FSDKaggle2018(**extraction_params,
                                  object_path=r'F:\Thesis_Results\Data_Readers\FSDKaggle2018')
        taskDatasets.append(fsdkaggle.taskDataset)
    if 4 in dataset_list:
        ravdess = Ravdess(**extraction_params,
                          object_path=r'F:\Thesis_Results\Data_Readers\Ravdess')
        taskDatasets.append(ravdess.taskDataset)
    if 5 in dataset_list:
        speechcommands = SpeechCommands(**extraction_params,
                                        object_path=r'F:\Thesis_Results\Data_Readers\SpeechCommands_{}')
        taskDatasets.append(speechcommands.taskDataset)
    if 6 in dataset_list:
        dcaseEvents = DCASE2017_SE(**extraction_params,
                                   object_path=r'F:\Thesis_Results\Data_Readers\DCASE2017_SE_{}')
        taskDatasets.append(dcaseEvents.taskDataset)
    print('loaded all datasets')
    for t in taskDatasets:
        t.save_split_scalers(**read_config('dic_of_labels_limits_{}'.format(t.task.name)), random_state=123)
        t.to_index_mode(**read_config('preparation_params_general_window'))
    print('done splitting')
    return taskDatasets


def main(argv):
    dataset_list = [0, 1, 2, 4, 5]

    print('--------------------------------------------------')
    print('write loop')
    print('--------------------------------------------------')
    for i in range(len(dataset_list)):
        run_datasets([dataset_list[i]], extraction_params=read_config('extraction_params_cnn_MelSpectrogram'))

    return 0


# tensorboard --logdir Tests/runs
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
# main(0)
