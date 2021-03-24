import torch
from torch import nn
import torch.nn.functional as F

from MultiTask.FFNN import FFNN
from MultiTask.TaskDependentLayers import TaskDependentLayers
from MultiTask.TaskDependentSoftmax import TaskDependentSoftmax
from pytorch_lightning.metrics.functional.classification import accuracy


class MultiTaskHardSharingConvolutional(nn.Module):

    def __init__(
            self,
            input_channels,
            hidden_size,  # 64
            n_hidden,
            task_list
    ):
        super().__init__()
        self.name = 'cnn'
        self.task_list = task_list

        self.hidden = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()
        for k in range(n_hidden):
            in_chan = input_channels if k == 0 else hidden_size
            self.hidden.append(
                nn.Conv2d(in_channels=in_chan,
                          out_channels=hidden_size,
                          kernel_size=(5, 5), stride=(1, 1),
                          padding=(1, 1))
            )
            self.hidden_bn.append(
                nn.BatchNorm2d(hidden_size)
            )

        for k in range(len(self.hidden)):
            torch.nn.init.kaiming_uniform_(self.hidden[k].weight)
            nn.init.constant(self.hidden[k].bias, 0.0)
            nn.init.normal_(self.hidden_bn[k].weight, 1.0, 0.02)
            nn.init.constant_(self.hidden_bn[k].bias, 0.0)

        self.task_nets = nn.ModuleList()
        for t in task_list:
            self.task_nets.append(
                nn.Linear(
                    hidden_size,
                    len(t.output_labels)
                )
            )

        for t in range(len(self.task_nets)):
            torch.nn.init.xavier_uniform_(self.task_nets[t].weight)
            nn.init.constant_(self.task_nets[t].bias, 0.0)

    def activate(self, x, activation):
        if activation == 'softmax':
            x = torch.softmax(x, dim=1)
            (x, _) = torch.max(x, dim=1)
            return x
        elif activation == 'sigmoid':
            x = torch.sigmoid(x)
            (x, _) = torch.max(x, dim=1)
            return x

    def forward(self, x):
        '''x: (batch_size, time_steps, mel_bins)'''
        x = x[None, :, :, :]
        x = torch.transpose(x, 0, 1)
        '''x: (batch_size, n_channels=1, time_steps, mel_bins)'''
        for layer_id in range(len(self.hidden)):
            x = F.relu_(self.hidden_bn[layer_id](self.hidden[layer_id](x)))
            x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        # Global max pooling
        (x, _) = torch.max(x, dim=3)
        '''x: (batch_size, feature_maps, time_steps)'''
        x = torch.transpose(x, 1, 2)
        '''x: (batch_size, time_steps, feature_maps)'''

        return tuple(self.activate(self.task_nets[task_model_id](x), self.task_list[task_model_id].output_module)
                     for task_model_id in range(len(self.task_list)))
