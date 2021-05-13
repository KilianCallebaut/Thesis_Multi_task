from __future__ import print_function, division

import sys

import torch

from DataReaders.ASVspoof2015 import ASVspoof2015
from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.DCASE2017_SS import DCASE2017_SS
from DataReaders.DCASE2017_SE import DCASE2017_SE
from DataReaders.FSDKaggle2018 import FSDKaggle2018
from DataReaders.Ravdess import Ravdess
from DataReaders.SpeechCommands import SpeechCommands
from MultiTask.MultiTaskHardSharingConvolutional import MultiTaskHardSharingConvolutional
from Tests.config_reader import *
from Training.Results import Results
from Training.Training import Training
from Tasks.ConcatTaskDataset import ConcatTaskDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'F'


def run_datasets(dataset_list):
    # extraction_params = read_config('extraction_params_cnn_MelSpectrogram')
    extraction_params = read_config('extraction_params_cnn_mfcc')
    calculate_window_size(extraction_params)
    taskdatasets = []
    evaldatasets = []

    if 0 in dataset_list:
        asvspoof = ASVspoof2015(**extraction_params,
                                object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\ASVspoof2015_{}')
        asvspoof.prepare_taskDatasets(**read_config('preparation_params_asvspoof_cnn'))
        asvspoof_t = asvspoof.toTrainTaskDataset()
        asvspoof_e = asvspoof.toValidTaskDataset()
        taskdatasets.append(asvspoof_t)
        evaldatasets.append(asvspoof_e)
    if 1 in dataset_list:
        chenaudio = ChenAudiosetDataset(**extraction_params,
                                        object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\ChenAudiosetDataset')
        chenaudio.prepare_taskDatasets(**read_config('preparation_params_chen_cnn'))
        chen_t = chenaudio.toTrainTaskDataset()
        chen_e = chenaudio.toTestTaskDataset()
        taskdatasets.append(chen_t)
        evaldatasets.append(chen_e)
    if 2 in dataset_list:
        dcaseScene = DCASE2017_SS(**extraction_params,
                                  object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SS_{}')
        dcaseScene.prepare_taskDatasets(**read_config('preparation_params_dcaseScene_cnn'))
        dcasScent_t = dcaseScene.toTrainTaskDataset()
        dcasScent_e = dcaseScene.toValidTaskDataset()
        taskdatasets.append(dcasScent_t)
        evaldatasets.append(dcasScent_e)
    if 3 in dataset_list:
        fsdkaggle = FSDKaggle2018(**extraction_params,
                                  object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\FSDKaggle2018')
        fsdkaggle.prepare_taskDatasets(**read_config('preparation_params_fsdkaggle_cnn'))
        fsdkaggle_t = fsdkaggle.toTrainTaskDataset()
        fsdkaggle_e = fsdkaggle.toTestTaskDataset()
        taskdatasets.append(fsdkaggle_t)
        evaldatasets.append(fsdkaggle_e)
    if 4 in dataset_list:
        ravdess = Ravdess(**extraction_params, object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\Ravdess')
        ravdess.prepare_taskDatasets(**read_config('preparation_params_ravdess_cnn'))
        ravdess_t = ravdess.toTrainTaskDataset()
        ravdess_e = ravdess.toTestTaskDataset()
        taskdatasets.append(ravdess_t)
        evaldatasets.append(ravdess_e)
    if 5 in dataset_list:
        speechcommands = SpeechCommands(**extraction_params,
                                        object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\SpeechCommands_{}')
        speechcommands.prepare_taskDatasets(**read_config('preparation_params_speechcommands_cnn'))
        speechcommands_t = speechcommands.toTrainTaskDataset()
        speechcommands_e = speechcommands.toValidTaskDataset()
        taskdatasets.append(speechcommands_t)
        evaldatasets.append(speechcommands_e)
    if 6 in dataset_list:
        dcaseEvents = DCASE2017_SE(**extraction_params,
                                  object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SE_{}')
        dcaseEvents.prepare_taskDatasets(**read_config('preparation_params_dcaseScene_cnn'))
        dcaseEvents_t = dcaseEvents.toTrainTaskDataset()
        dcaseEvents_e = dcaseEvents.toValidTaskDataset()
        taskdatasets.append(dcaseEvents_t)
        evaldatasets.append(dcaseEvents_e)
    print('loaded all datasets')

    return taskdatasets, evaldatasets


def run_test(eval_dataset, meta_params, run_name, **kwargs):
    results = Results.create_model_loader(run_name, **kwargs)
    task_list = eval_dataset.get_task_list()
    blank_model = MultiTaskHardSharingConvolutional(1,
                                                    **read_config('model_params_cnn'),
                                                    task_list=task_list)
    blank_model = blank_model.to(device)
    Training.evaluate(blank_model=blank_model,
                      concat_dataset=eval_dataset,
                      training_results=results,
                      batch_size=meta_params['batch_size'],
                      num_epochs=meta_params['num_epochs'])


def get_concat(dataset_list):
    taskdatasets, evaldatasets = run_datasets(dataset_list)
    training_dataset = ConcatTaskDataset(taskdatasets)
    eval_dataset = ConcatTaskDataset(evaldatasets)
    return training_dataset, eval_dataset


def main(argv):
    model_checkpoints_path = drive + r":\Thesis_Results\Model_Checkpoints"
    meta_params = read_config('meta_params_cnn_MelSpectrogram')
    dataset_list = [0, 1, 2, 4, 5]
    run_datasets(dataset_list)

    print('--------------------------------------------------')
    print('test loop')
    for i in range(len(dataset_list)):
        training_dataset, eval_dataset = get_concat([dataset_list[i]])
        task_list = training_dataset.get_task_list()
        print(task_list[0].name)
        model = MultiTaskHardSharingConvolutional(1,
                                                  **read_config('model_params_cnn'),
                                                  task_list=task_list)
        model = model.to(device)
        print('Model Created')
        results = Results(model_checkpoints_path=model_checkpoints_path)
        model, results = Training.run_gradient_descent(model=model,
                                                       concat_dataset=training_dataset,
                                                       results=results,
                                                       batch_size=meta_params['batch_size'],
                                                       num_epochs=meta_params['num_epochs'],
                                                       learning_rate=meta_params['learning_rate'])

        run_name = results.run_name
        del model, results
        torch.cuda.empty_cache()
        run_test(eval_dataset, meta_params, run_name)

        for j in range(i + 1, len(dataset_list)):
            training_dataset, eval_dataset = get_concat([dataset_list[i], dataset_list[j]])
            task_list = training_dataset.get_task_list()
            print(task_list[0].name + ' combined with ' + task_list[1].name)
            model = MultiTaskHardSharingConvolutional(1,
                                                      **read_config('model_params_cnn'),
                                                      task_list=task_list)
            model = model.to(device)
            print('Model Created')
            results = Results(model_checkpoints_path=model_checkpoints_path)
            model, results = Training.run_gradient_descent(model=model,
                                                           concat_dataset=training_dataset,
                                                           results=results,
                                                           batch_size=meta_params['batch_size'],
                                                           num_epochs=meta_params['num_epochs'],
                                                           learning_rate=meta_params['learning_rate'])
            run_name = results.run_name
            del model, results
            torch.cuda.empty_cache()
            run_test(eval_dataset, meta_params, run_name, model_checkpoints_path=model_checkpoints_path)

    return 0


# tensorboard --logdir Tests/runs
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
# main(0)
