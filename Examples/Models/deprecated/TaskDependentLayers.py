import torch
from torch import nn
import torch.nn.functional as F

from MultiTask.FFNN import FFNN


class TaskDependentLayers(nn.Module):

    def __init__(
            self,
            hidden_size,
            task_list,
    ):
        super().__init__()
        self.taskList = task_list
        self.task_nets = nn.ModuleList()
        for t in task_list:
            self.task_nets.append(
                nn.Linear(
                    hidden_size,
                    len(t.output_labels)
                )
            )

    def activate(self, x, activation):
        if activation == 'softmax':
            return torch.softmax(x, dim=1)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)

    def forward(self, x):
        return torch.cat(
            tuple(self.activate(self.task_nets[task_model_id](x), self.taskList[task_model_id].output_module)
                  for task_model_id in range(len(self.taskList))),
            dim=1
        )
