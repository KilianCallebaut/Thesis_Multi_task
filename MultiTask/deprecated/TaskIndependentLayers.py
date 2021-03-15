import torch
from torch import nn
from MultiTask.FFNN import FFNN


class TaskIndependentLayers(nn.Module):
    """NN for MTL with hard parameter sharing
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            n_outputs,
            dropout_rate=.1,
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.task_nets = nn.ModuleList()
        for _ in range(n_outputs):
            self.task_nets.append(
                FFNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_hidden=n_hidden,
                    n_outputs=1,
                    dropout_rate=dropout_rate,
                )
            )

    def forward(self, x):
        return torch.cat(
            tuple(task_model(x) for task_model in self.task_nets),
            dim=1
        )
