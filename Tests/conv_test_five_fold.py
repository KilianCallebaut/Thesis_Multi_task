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
from Tasks.TrainingSetCreator import ConcatTrainingSetCreator
from Tests.config_reader import *
from Training.Results import Results
from Training.Training import Training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'F'
# data_base = r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers'
data_base = r'F:\Thesis_Results\Data_Readers'


def run_five_fold(dataset_list, **kwargs):
    extraction_params = read_config('extraction_params_cnn_LibMelSpectrogram')
    # extraction_params = read_config('extraction_params_cnn_mfcc')
    taskDatasets, testDatasets = run_datasets(dataset_list, extraction_params)
    print("Start iteration")
    i = 0
    ctsc = ConcatTrainingSetCreator(training_sets=taskDatasets,
                                    dics_of_labels_limits=[read_config('dic_of_labels_limits_{}'.format(t.task.name))[
                                                               'dic_of_labels_limits'] for t in taskDatasets],
                                    random_state=123,
                                    test_sets=testDatasets)
    for train, test in ctsc.generate_concats():
        if 'fold' in kwargs and kwargs.get('fold') > i:
            i += 1
            continue
        run_set(train, test, i)
        i += 1


def run_datasets(dataset_list, extraction_params):
    taskDatasets = []
    testDatasets = []

    if 0 in dataset_list:
        asvspoof = ASVspoof2015(**extraction_params,
                                object_path=os.path.join(data_base, 'ASVspoof2015_{}'))
        taskDatasets.append(asvspoof.taskDataset)
    if 1 in dataset_list:
        chenaudio = ChenAudiosetDataset(**extraction_params,
                                        object_path=os.path.join(data_base, 'ChenAudiosetDataset'))
        taskDatasets.append(chenaudio.taskDataset)
    if 2 in dataset_list:
        dcaseScene = DCASE2017_SS(**extraction_params,
                                  object_path=os.path.join(data_base, 'DCASE2017_SS_{}'))
        taskDatasets.append(dcaseScene.taskDataset)
        testDatasets.append(dcaseScene.valTaskDataset)
    if 3 in dataset_list:
        fsdkaggle = FSDKaggle2018(**extraction_params,
                                  object_path=os.path.join(data_base, 'FSDKaggle2018_{}'))
        taskDatasets.append(fsdkaggle.taskDataset)
    if 4 in dataset_list:
        ravdess = Ravdess(**extraction_params,
                          object_path=os.path.join(data_base, 'Ravdess'))
        taskDatasets.append(ravdess.taskDataset)
    if 5 in dataset_list:
        speechcommands = SpeechCommands(**extraction_params,
                                        object_path=os.path.join(data_base, 'SpeechCommands_{}'))
        taskDatasets.append(speechcommands.taskDataset)
        testDatasets.append(speechcommands.validTaskDataset)
    if 6 in dataset_list:
        dcaseEvents = DCASE2017_SE(**extraction_params,
                                   object_path=os.path.join(data_base, 'DCASE2017_SE_{}'))
        taskDatasets.append(dcaseEvents.taskDataset)
    print('loaded all datasets')

    return taskDatasets, testDatasets


def run_set(concat_training, concat_test, fold):
    model_checkpoints_path = drive + r":\Thesis_Results\Model_Checkpoints"
    meta_params = read_config('meta_params_cnn_MelSpectrogram')

    task_list = concat_training.get_task_list()

    model = MultiTaskHardSharingConvolutional(1,
                                              **read_config('model_params_cnn'),
                                              task_list=task_list)
    # model = BaselineCnn(n_hidden=4,
    #                     task_list=task_list,
    #                     drop_rate=0)

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
                                                   learning_rate=meta_params['learning_rate'],
                                                   test_dataset=concat_test)

    del model
    torch.cuda.empty_cache()
    # run_test(concat_test, meta_params, results)
    results.write_loss_curve_tasks()
    results.write_loss_curves()
    results.close_writer()


def sample_datasets(datasets):
    sampled_datasets = []
    for d in datasets:
        sampled_datasets.append(d.sample_dataset(random_state=123))
    return sampled_datasets


def create_index_mode(dataset_list):
    extraction_params = read_config('extraction_params_cnn_LibMelSpectrogram')
    # extraction_params = read_config('extraction_params_cnn_mfcc')
    taskDatasets, testDatasets = run_datasets(dataset_list, extraction_params)
    print("Start create index mode")
    ctsc = ConcatTrainingSetCreator(training_sets=taskDatasets,
                                    dics_of_labels_limits=[read_config('dic_of_labels_limits_{}'.format(t.task.name))[
                                                               'dic_of_labels_limits'] for t in taskDatasets],
                                    random_state=123,
                                    test_sets=testDatasets)
    ctsc.prepare_scalers()
    print('writing to index mode')
    ctsc.prepare_for_index_mode()


def main(argv):
    # dataset_list = [2, 5, 4, 1, 0]
    dataset_list = [5]
    dataset_list_single = [2, 5, 4, 1, 0]
    # dataset_list_double = [[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [1, 4], [1, 5], [2, 4], [2, 5], [4, 5]]
    dataset_list_double = [[0, 1], [0, 2], [0, 4], [0, 5], [1, 2], [1, 4], [1, 5], [2, 4], [2, 5], [4, 5]]

    print('--------------------------------------------------')
    print('test loop')
    print('--------------------------------------------------')

    # for i in dataset_list:
    #     create_index_mode([i])

    for i in dataset_list_single:
        # create_index_mode([2])
        # create_index_mode([i])
        run_five_fold([i])
    #     # for j in range(i + 1, len(dataset_list)):
    #     #     #     # check_distributions([dataset_list[i], dataset_list[j]])
    #     #     run_five_fold([dataset_list[i], dataset_list[j]], fold=4)
    #     # # # check_distributions([dataset_list[i]])
    #
    for i in dataset_list_double:
        run_five_fold(i)

    return 0


# tensorboard --logdir F:\Thesis_Results\Training_Results\experiments
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
