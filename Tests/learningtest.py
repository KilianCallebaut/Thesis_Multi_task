from __future__ import print_function, division

import torch
from torch import nn
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image

from DataReaders.ChenAudiosetDataset import ChenAudiosetDataset
from DataReaders.Test_fgo import Test_fgo
from MultiTask.HardSharing import HardSharing
from MultiTask.pytorch_multitask_example.multi_output_model import multi_output_model
from Tasks.ConcatTaskDataset import ConcatTaskDataset
from Tasks.Task import Task
from Tasks.TaskDataset import TaskDataset
from Training.Training import Training

plt.ion()  # interactive mode
import pandas as pd
from sklearn.model_selection import train_test_split

# own model
# Important: the tasks in task_list has to be the same as the order in concatTaskDataset
# Required:
# ConcatTaskDataset
# ->    list of TaskDataset
# ->    inputs: list of size nr_instances, flatened tensors
# ->    targets: tensors of right outputs
# ->    name: name as task, consistent with task_list
# X_input_size: input nodes amount
# task_list: list of object Task:
# ->    name: name as task, consistent with concattaskdataset
# ->    output_labels: list of string with output labels

# test_fgo = Test_fgo()
# training_dataset = test_fgo.toTaskDataset()
#
# model = HardSharing(test_fgo.get_input_nodes(), 128, 3, test_fgo.get_task_list())
# model = model.cuda()

chendata = ChenAudiosetDataset()

chenTaskDataset = chendata.toTaskDataset()
training_dataset = ConcatTaskDataset([chenTaskDataset])
task_list = training_dataset.get_task_list()

model = HardSharing(len(chenTaskDataset.inputs[0]), 128, 3, task_list)
model = model.cuda()



print('started')
Training.run_gradient_descent(
    model=model,
    concat_dataset=training_dataset,
    batch_size=16,
    num_epochs=20
    # learning_rate=,
    # weight_decay=,
)


train_loader = torch.utils.data.DataLoader(
            training_dataset,
            # sampler=BalancedBatchSchedulerSampler(dataset=concat_dataset,
            #                                       batch_size=batch_size),
            batch_size=1,
            shuffle=True,
            num_workers=0)

model.eval()
outputs = []
trues = []
for inputs, targets, names in train_loader:
    out = model(inputs.cuda())
    outputs.append(out)
    true = targets
    trues.append(true)
outputs = torch.cat(outputs, dim=0)
trues = torch.cat(trues, dim=0)


a = Training.calculate_stats(outputs.detach().cpu(), trues.detach().cpu())
print(a)


[[t.output_labels] for t in task_list]