from __future__ import print_function, division

import sys

import torch

from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.DCASE2017_SS import DCASE2017_SS
from DataReaders.FSDKaggle2018 import FSDKaggle2018
from DataReaders.Ravdess import Ravdess
from DataReaders.SpeechCommands import SpeechCommands
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from Tests.config_reader import *
from Training.Results import Results
from Training.Training import Training
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tests.conv_test import *

meta_params = read_config('meta_params_cnn_MelSpectrogram')
model_checkpoints_path = r"D:\Thesis_Results\Model_Checkpoints"
dataset_list = [0, 2]
training_dataset, eval_dataset = get_concat([dataset_list[0], dataset_list[1]])
task_list = training_dataset.get_task_list()
results = Results(run_name=r'Result_20_04_2021_18_10_40', model_checkpoints_path=model_checkpoints_path)
model = MultiTaskHardSharingConvolutional(1, **read_config('model_params_cnn'),
                                          task_list=task_list)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
results.load_model_parameters(52, model)
model, results = Training.run_gradient_descent(model=model,
                                               concat_dataset=training_dataset,
                                               results=results,
                                               batch_size=meta_params['batch_size'],
                                               num_epochs=meta_params['num_epochs'],
                                               learning_rate=meta_params['learning_rate'],
                                               start_epoch=52)
run_name = results.run_name
del model, results
torch.cuda.empty_cache()
run_test(eval_dataset, meta_params, run_name, model_checkpoints_path=model_checkpoints_path)

