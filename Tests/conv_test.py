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
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tests.config_reader import *
from Training.Training import Training
from Training.Results import Results


def run_datasets(dataset_list):
    extraction_params = read_config('extraction_params_cnn_MelSpectrogram')
    calculate_window_size(extraction_params)
    taskdatasets = []
    evaldatasets = []

    if 0 in dataset_list:
        asvspoof = ASVspoof2015(**extraction_params)
        asvspoof.prepare_taskDatasets(**read_config('preparation_params_asvspoof_cnn'))
        asvspoof_t = asvspoof.toTrainTaskDataset()
        asvspoof_e = asvspoof.toValidTaskDataset()
        taskdatasets.append(asvspoof_t)
        evaldatasets.append(asvspoof_e)
    if 1 in dataset_list:
        chenaudio = ChenAudiosetDataset(**extraction_params)
        chenaudio.prepare_taskDatasets(**read_config('preparation_params_chen_cnn'))
        chen_t = chenaudio.toTrainTaskDataset()
        chen_e = chenaudio.toTestTaskDataset()
        taskdatasets.append(chen_t)
        evaldatasets.append(chen_e)
    if 2 in dataset_list:
        dcaseScene = DCASE2017_SS(**extraction_params)
        dcaseScene.prepare_taskDatasets(**read_config('preparation_params_dcaseScene_cnn'))
        dcasScent_t = dcaseScene.toTrainTaskDataset()
        dcasScent_e = dcaseScene.toValidTaskDataset()
        taskdatasets.append(dcasScent_t)
        evaldatasets.append(dcasScent_e)
    if 3 in dataset_list:
        fsdkaggle = FSDKaggle2018(**extraction_params)
        fsdkaggle.prepare_taskDatasets(**read_config('preparation_params_fsdkaggle_cnn'))
        fsdkaggle_t = fsdkaggle.toTrainTaskDataset()
        fsdkaggle_e = fsdkaggle.toTestTaskDataset()
        taskdatasets.append(fsdkaggle_t)
        evaldatasets.append(fsdkaggle_e)
    if 4 in dataset_list:
        ravdess = Ravdess(**extraction_params)
        ravdess.prepare_taskDatasets(**read_config('preparation_params_ravdess_cnn'))
        ravdess_t = ravdess.toTrainTaskDataset()
        ravdess_e = ravdess.toTestTaskDataset()
        taskdatasets.append(ravdess_t)
        evaldatasets.append(ravdess_e)
    if 5 in dataset_list:
        speechcommands = SpeechCommands(**extraction_params)
        speechcommands.prepare_taskDatasets(**read_config('preparation_params_speechcommands_cnn'))
        speechcommands_t = speechcommands.toTrainTaskDataset()
        speechcommands_e = speechcommands.toValidTaskDataset()
        taskdatasets.append(speechcommands_t)
        evaldatasets.append(speechcommands_e)
    print('loaded all datasets')

    return taskdatasets, evaldatasets


def run_test(eval_dataset, meta_params, run_name):
    results = Results.create_model_loader(run_name)
    task_list = eval_dataset.get_task_list()
    blank_model = MultiTaskHardSharingConvolutional(1,
                                                    **read_config('model_params_cnn'),
                                                    task_list=task_list)
    blank_model = blank_model.cuda()
    Training.evaluate(blank_model=blank_model,
                      concat_dataset=eval_dataset,
                      training_results=results,
                      batch_size=meta_params['batch_size'],
                      num_epochs=meta_params['num_epochs'])


def main(argv):
    extraction_params = read_config('extraction_params_cnn_MelSpectrogram')
    calculate_window_size(extraction_params)
    meta_params = read_config('meta_params_cnn_MelSpectrogram')
    # dataset_list = [0, 1, 2, 4]
    dataset_list = [0, 2]
    # dataset_list = [2]
    taskdatasets, evaldatasets = run_datasets(dataset_list)

    print('--------------------------------------------------')
    print('test loop')
    for i in range(len(taskdatasets)):
        # print(taskdatasets[i].task.name)
        # training_dataset = ConcatTaskDataset([taskdatasets[i]])
        # eval_dataset = ConcatTaskDataset([evaldatasets[i]])
        # task_list = training_dataset.get_task_list()
        # model = MultiTaskHardSharingConvolutional(1,
        #                                           **read_config('model_params_cnn'),
        #                                           task_list=task_list)
        # model = model.cuda()
        # print('Model Created')
        #
        # model, results = Training.run_gradient_descent(model=model,
        #                                                concat_dataset=training_dataset,
        #                                                batch_size=meta_params['batch_size'],
        #                                                num_epochs=meta_params['num_epochs'],
        #                                                learning_rate=meta_params['learning_rate'])
        #
        # run_name = results.run_name
        # del model, results
        # torch.cuda.empty_cache()
        # run_test(eval_dataset, meta_params, run_name)

        for j in range(i + 1, len(taskdatasets)):
            print(taskdatasets[i].task.name + ' combined with ' + taskdatasets[j].task.name)
            training_dataset = ConcatTaskDataset([taskdatasets[i], taskdatasets[j]])
            eval_dataset = ConcatTaskDataset([evaldatasets[i], evaldatasets[j]])
            task_list = training_dataset.get_task_list()
            model = MultiTaskHardSharingConvolutional(1,
                                                      **read_config('model_params_cnn'),
                                                      task_list=task_list)
            model = model.cuda()
            print('Model Created')

            model, results = Training.run_gradient_descent(model=model,
                                                           concat_dataset=training_dataset,
                                                           batch_size=meta_params['batch_size'],
                                                           num_epochs=meta_params['num_epochs'],
                                                           learning_rate=meta_params['learning_rate'])
            run_name = results.run_name
            del model, results
            torch.cuda.empty_cache()
            run_test(eval_dataset, meta_params, run_name)

    return 0


# tensorboard --logdir Tests/runs
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
# main(0)