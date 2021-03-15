from __future__ import print_function, division

import sys

from DataReaders.DCASE2017_SE import DCASE2017_SE

import torch
from torch import nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image

from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.DCASE2017_SS import DCASE2017_SS
from DataReaders.Test_fgo import Test_fgo
from MultiTask.HardSharing import HardSharing
from MultiTask.pytorch_multitask_example.multi_output_model import multi_output_model
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tasks.Task import Task
from Tasks.TaskDataset import TaskDataset
from Training.Training import Training


def main(argv):
    # dcasedata = DCASE2017_SE()
    # chenTaskDataset = dcasedata.toTaskDataset()

    chen_audio = ChenAudiosetDataset()
    chen_t = chen_audio.toTaskDataset()
    training_dataset = ConcatTaskDataset([chen_t])
    task_list = training_dataset.get_task_list()

    model = HardSharing(len(chen_t.inputs[0]), 128, 3, task_list)
    model = model.cuda()
    print('Model Created')

    model, results = Training.run_gradient_descent(model=model,
                                                   concat_dataset=training_dataset,
                                                   batch_size=16,
                                                   num_epochs=200,
                                                   learning_rate=0.001)

    results.plot_training_curve_loss_overall()
    print(results.calculate_metrics_epoch_per_task(-1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

