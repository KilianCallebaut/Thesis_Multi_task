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
                                object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\ASVspoof2015_{}')
        taskDatasets.append(asvspoof.taskDataset)
    if 1 in dataset_list:
        chenaudio = ChenAudiosetDataset(**extraction_params,
                                        object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\ChenAudiosetDataset')
        taskDatasets.append(chenaudio.taskDataset)
    if 2 in dataset_list:
        dcaseScene = DCASE2017_SS(**extraction_params,
                                  object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SS_{}')
        taskDatasets.append(dcaseScene.taskDataset)
    if 3 in dataset_list:
        fsdkaggle = FSDKaggle2018(**extraction_params,
                                  object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\FSDKaggle2018')
        taskDatasets.append(fsdkaggle.taskDataset)
    if 4 in dataset_list:
        ravdess = Ravdess(**extraction_params,
                          object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\Ravdess')
        taskDatasets.append(ravdess.taskDataset)
    if 5 in dataset_list:
        speechcommands = SpeechCommands(**extraction_params,
                                        object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\SpeechCommands_{}')
        taskDatasets.append(speechcommands.taskDataset)
    if 6 in dataset_list:
        dcaseEvents = DCASE2017_SE(**extraction_params,
                                   object_path=r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers\DCASE2017_SE_{}')
        taskDatasets.append(dcaseEvents.taskDataset)
    print('loaded all datasets')

    return taskDatasets


def run_test(eval_dataset, meta_params, results):
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


def run_five_fold(dataset_list):
    extraction_params = read_config('extraction_params_cnn_MelSpectrogram')
    # extraction_params = read_config('extraction_params_cnn_mfcc')
    taskDatasets = run_datasets(dataset_list, extraction_params)
    task_iterators = []

    print("Create iterators")
    for t in taskDatasets:
        task_iterators.append(t.k_folds(**read_config('dic_of_labels_limits_{}'.format(t.task.name)), random_state=123))

    print("Start iteration")
    i = 0
    for train_index, test_index in task_iterators[0]:
        training_tasks = []
        test_tasks = []
        train, test = taskDatasets[0].get_split_by_index(train_index, test_index,
                                                         **read_config('preparation_params_general_window'))
        training_tasks.append(train)
        test_tasks.append(test)
        if len(task_iterators) > 1:
            for it_id in range(len(task_iterators[1:])):
                it = task_iterators[it_id + 1]
                train_nxt_id, test_nxt_id = next(it)
                train, test = taskDatasets[it_id + 1].get_split_by_index(train_nxt_id, test_nxt_id,
                                                                         **read_config(
                                                                             'preparation_params_general_window'))
                training_tasks.append(train)
                test_tasks.append(test)

        concat_training = ConcatTaskDataset(training_tasks)
        concat_test = ConcatTaskDataset(test_tasks)

        # Run multi task
        run_set(concat_training, concat_test, i)

        i += 1


def run_set(concat_training, concat_test, fold):
    model_checkpoints_path = drive + r":\Thesis_Results\Model_Checkpoints"
    meta_params = read_config('meta_params_cnn_MelSpectrogram')

    task_list = concat_training.get_task_list()
    model = MultiTaskHardSharingConvolutional(1,
                                              **read_config('model_params_cnn'),
                                              task_list=task_list)
    model = model.to(device)
    print('Model Created')

    # run_name creation
    run_name = "Result_" + str(
        datetime.now().strftime("%d_%m_%Y_%H_%M_%S")) + "_" + model.name
    for n in task_list:
        run_name += "_" + n.name
    run_name += "_fold_{}".format(fold)

    results = Results(model_checkpoints_path=model_checkpoints_path,
                      run_name=run_name, num_epochs=meta_params['num_epochs'])
    model, results = Training.run_gradient_descent(model=model,
                                                   concat_dataset=concat_training,
                                                   results=results,
                                                   batch_size=meta_params['batch_size'],
                                                   num_epochs=meta_params['num_epochs'],
                                                   learning_rate=meta_params['learning_rate'])

    del model
    torch.cuda.empty_cache()
    run_test(concat_test, meta_params, results)


def main(argv):
    dataset_list = [2, 5, 4, 0, 1]

    print('--------------------------------------------------')
    print('test loop')
    print('--------------------------------------------------')
    for i in range(len(dataset_list)):
        for j in range(i + 1, len(dataset_list)):
            run_five_fold([dataset_list[i], dataset_list[j]])
        run_five_fold([dataset_list[i]])

    return 0


# tensorboard --logdir Tests/runs
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
# main(0)
