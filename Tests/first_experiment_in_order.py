from __future__ import print_function, division

import sys

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
from Training.Training import Training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drive = 'F'
# data_base = r'C:\Users\mrKC1\PycharmProjects\Thesis\data\Data_Readers'
data_base = r'F:\Thesis_Results\Data_Readers'



# def create_index_mode(dataset_list):
#     taskDatasets = run_datasets(dataset_list)
#     print("Start create index mode")
#     ctsc = ConcatTrainingSetCreator(training_sets=taskDatasets,
#                                     dics_of_labels_limits=[read_config('dic_of_labels_limits_{}'.format(t.task.name))[
#                                                                'dic_of_labels_limits'] for t in taskDatasets],
#                                     random_state=123)
#     print('writing to index mode')
#     ctsc.prepare_for_index_mode()


def main(argv):
    print('--------------------------------------------------')
    print('test loop')
    print('--------------------------------------------------')
    taskDatasets = []
    extraction_params = read_config('extraction_params_cnn_LibMelSpectrogram')
    meta_params = read_config('meta_params_cnn_MelSpectrogram')

    ## Data Reading
    asvspoof = ASVspoof2015(**extraction_params,
                            object_path=os.path.join(data_base, 'ASVspoof2015_{}'))
    taskDatasets.append(asvspoof.taskDataset)

    chenaudio = ChenAudiosetDataset(**extraction_params,
                                    object_path=os.path.join(data_base, 'ChenAudiosetDataset'))
    taskDatasets.append(chenaudio.taskDataset)

    dcaseScene = DCASE2017_SS(**extraction_params,
                              object_path=os.path.join(data_base, 'DCASE2017_SS_{}'))
    taskDatasets.append(dcaseScene.taskDataset)
    fsdkaggle = FSDKaggle2018(**extraction_params,
                              object_path=os.path.join(data_base, 'FSDKaggle2018_{}'))
    taskDatasets.append(fsdkaggle.taskDataset)
    ravdess = Ravdess(**extraction_params,
                      object_path=os.path.join(data_base, 'Ravdess'))
    taskDatasets.append(ravdess.taskDataset)
    speechcommands = SpeechCommands(**extraction_params,
                                    object_path=os.path.join(data_base, 'SpeechCommands_{}'))
    taskDatasets.append(speechcommands.taskDataset)
    dcaseEvents = DCASE2017_SE(**extraction_params,
                               object_path=os.path.join(data_base, 'DCASE2017_SE_{}'))
    taskDatasets.append(dcaseEvents.taskDataset)
    print('loaded all datasets')

    #### Data Loading
    print("Start iteration")
    i = 0
    ctsc = ConcatTrainingSetCreator(training_sets=taskDatasets,
                                    dics_of_labels_limits=[read_config('dic_of_labels_limits_{}'.format(t.task.name))[
                                                               'dic_of_labels_limits'] for t in taskDatasets],
                                    random_state=123)
    for train, test in ctsc.generate_concats():
        if 'fold' in kwargs and kwargs.get('fold') > i:
            i += 1
            continue
        #### Training
        #################################################################################
        task_list = train.get_task_list()

        model = MultiTaskHardSharingConvolutional(1,
                                                  **read_config('model_params_cnn'),
                                                  task_list=task_list)
        model = model.to(device)
        print('Model Created')

        #################################################################################
        # run_name creation
        results = Training.create_results(modelname=model.name,
                                          task_list=task_list,
                                          fold=i,
                                          model_checkpoints_path=drive + r":\Thesis_Results\Model_Checkpoints",
                                          num_epochs=meta_params['num_epochs'])

        #################################################################################

        model, results = Training.run_gradient_descent(model=model,
                                                       concat_dataset=train,
                                                       results=results,
                                                       batch_size=meta_params['batch_size'],
                                                       num_epochs=meta_params['num_epochs'],
                                                       learning_rate=meta_params['learning_rate'],
                                                       test_dataset=test)

        results.write_loss_curve_tasks()
        results.write_loss_curves()
        results.close_writer()

        #################################################################################
        i += 1

    return 0


# tensorboard --logdir F:\Thesis_Results\Training_Results\experiments
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
