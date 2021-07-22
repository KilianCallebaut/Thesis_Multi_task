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
    extraction_params = read_config('extraction_params_cnn_LibMelSpectrogram')
    meta_params = read_config('meta_params_cnn_MelSpectrogram')

    ## Data Reading
    asvspoof = ASVspoof2015(**extraction_params,
                            object_path=os.path.join(data_base, 'ASVspoof2015_{}'))
    chenaudio = ChenAudiosetDataset(**extraction_params,
                                    object_path=os.path.join(data_base, 'ChenAudiosetDataset'))
    dcaseScene = DCASE2017_SS(**extraction_params,
                              object_path=os.path.join(data_base, 'DCASE2017_SS_{}'))
    fsdkaggle = FSDKaggle2018(**extraction_params,
                              object_path=os.path.join(data_base, 'FSDKaggle2018_{}'))
    ravdess = Ravdess(**extraction_params,
                      object_path=os.path.join(data_base, 'Ravdess'))
    speechcommands = SpeechCommands(**extraction_params,
                                    object_path=os.path.join(data_base, 'SpeechCommands_{}'))
    dcaseEvents = DCASE2017_SE(**extraction_params,
                               object_path=os.path.join(data_base, 'DCASE2017_SE_{}'))
    print('loaded all datasets')

    #### Data Loading
    print("Start iteration")
    i = 0
    ctsc = ConcatTrainingSetCreator(random_state=123)
    asvspoof_task = asvspoof.return_taskDataset()
    ctsc.add_dataset(dataset=asvspoof_task,
                     dic_of_labels_limits=read_config('dic_of_labels_limits_{}'.format(asvspoof_task.task.name))[
                         'dic_of_labels_limits'])
    chen_task = chenaudio.return_taskDataset()
    ctsc.add_dataset(dataset=chen_task,
                     dic_of_labels_limits=read_config('dic_of_labels_limits_{}'.format(chen_task.task.name))[
                         'dic_of_labels_limits'])
    dcaseScene_task = dcaseScene.return_taskDataset()
    ctsc.add_dataset(dataset=dcaseScene_task,
                     dic_of_labels_limits=read_config('dic_of_labels_limits_{}'.format(dcaseScene_task.task.name))[
                         'dic_of_labels_limits'])
    fsdkaggle_task = fsdkaggle.return_taskDataset()
    ctsc.add_dataset(dataset=fsdkaggle_task,
                     dic_of_labels_limits=read_config('dic_of_labels_limits_{}'.format(fsdkaggle_task.task.name))[
                         'dic_of_labels_limits'])
    ravdess_task = ravdess.return_taskDataset()
    ctsc.add_dataset(dataset=ravdess_task,
                     dic_of_labels_limits=read_config('dic_of_labels_limits_{}'.format(ravdess_task.task.name))[
                         'dic_of_labels_limits'])
    speechcommands_task = speechcommands.return_taskDataset()
    ctsc.add_dataset(dataset=speechcommands_task,
                     dic_of_labels_limits=read_config('dic_of_labels_limits_{}'.format(speechcommands_task.task.name))[
                         'dic_of_labels_limits'])
    dcaseEvents_task = dcaseEvents.return_taskDataset()
    ctsc.add_dataset(dataset=dcaseEvents_task,
                     dic_of_labels_limits=read_config('dic_of_labels_limits_{}'.format(dcaseEvents_task.task.name))[
                         'dic_of_labels_limits']
                     )

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
