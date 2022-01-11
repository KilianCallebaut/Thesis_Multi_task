import torch
from torch import nn
from MultiTask.FFNN import FFNN


class TaskDependentSoftmax(nn.Module):

    def __init__(
            self,
            hidden_size,
            task_list
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return torch.cat(
            tuple(self.softmax(task_model(x)) for task_model in self.task_nets),
            dim=1
        )
