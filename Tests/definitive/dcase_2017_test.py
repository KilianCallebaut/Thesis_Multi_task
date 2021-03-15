from __future__ import print_function, division

import sys

from DataReaders.DCASE2017_SS import DCASE2017_SS
from MultiTask.MultiTaskHardSharing import MultiTaskHardSharing
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Training.Training import Training


def main(argv):
    meta_params = {
        'extraction_method': 'logbank_summary',
        'resample_to': 8000,
        'batch_size': 8,
        'num_epochs': 500,
        'learning_rate': 0.001
    }

    extraction_params = dict(
        winlen=0.03,
        winstep=0.01,
        nfilt=24,

        nfft=1024,  # 512,
        lowfreq=0,
        highfreq=None,
        preemph=0  # 0.97
    )

    model_params = dict(
        hidden_size=512,
        n_hidden=4
    )

    test_size = 0

    dcaseScene = DCASE2017_SS(extraction_method=meta_params['extraction_method'],
                              test_size=test_size,
                              **extraction_params,
                              resample_to=meta_params['resample_to'])

    dcasScent_t = dcaseScene.toTrainTaskDataset()
    taskdatasets = [dcasScent_t]

    dcasScent_e = dcaseScene.toValidTaskDataset()
    evaldatasets = [dcasScent_e]
    print('Done loading')

    print('test loop')
    for i in range(len(taskdatasets)):
        print(taskdatasets[i].task.name)
        training_dataset = ConcatTaskDataset([taskdatasets[i]])
        eval_dataset = ConcatTaskDataset([evaldatasets[i]])
        task_list = training_dataset.get_task_list()
        model = MultiTaskHardSharing(taskdatasets[i].inputs[0].shape[0], **model_params, task_list=task_list)
        model = model.cuda()
        print('Model Created')

        model, results = Training.run_gradient_descent(model=model,
                                                       concat_dataset=training_dataset,
                                                       batch_size=meta_params['batch_size'],
                                                       num_epochs=meta_params['num_epochs'],
                                                       learning_rate=meta_params['learning_rate'])
        Training.evaluate(blank_model=model,
                          concat_dataset=eval_dataset,
                          training_results=results,
                          batch_size=meta_params['batch_size'],
                          num_epochs=meta_params['num_epochs'])

    return 0


# tensorboard --logdir Tests/runs
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
