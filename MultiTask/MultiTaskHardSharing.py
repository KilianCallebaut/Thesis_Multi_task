import torch
from torch import nn
import torch.nn.functional as F


# https://gist.github.com/AvivNavon/cf2071ffaadfb11dded915f7f4bd638e
# https://medium.com/@adrian.waelchli/3-simple-tricks-that-will-change-the-way-you-debug-pytorch-5c940aa68b03
class MultiTaskHardSharing(nn.Module):

    # Hyperparameters for our network
    # Input: 7 (summaries) x # coefficients (typically 24 for mel-scaled filterbanks)
    # 1 hidden layer with 128 units
    # ReLU activation
    # 1 hidden layer with 128 units
    # Another ReLU
    # 1 hidden layer with 128 units
    # Another ReLU
    # Output layer with softmax (task dependent)
    #
    # input_size
    # hidden_size: number of nodes in all hidden layers
    # n_hidden: number of hidden layers in shared NN
    # task_list: list of task objects
    # output_tasks: list of output tasks
    # n_task_specific_layers: number of layers that are task specific, should be 0
    # task_specific_hidden_size: number of nodes in task specific hidden layers
    # dropout_rate
    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            task_list
    ):
        super().__init__()
        self.name = 'dnn'
        self.task_list = task_list
        # self.model = nn.Sequential()

        h_sizes = [input_size] + [hidden_size for _ in range(n_hidden)]

        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(
                nn.Linear(
                    h_sizes[k],
                    h_sizes[k + 1]
                )
            )
        for k in range(len(self.hidden)):
            torch.nn.init.xavier_uniform_(self.hidden[k].weight)
            nn.init.constant_(self.hidden[k].bias, 0)

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
            nn.init.constant_(self.task_nets[t].bias, 0)

    def activate(self, x, activation):
        if activation == 'softmax':
            return F.softmax(x, dim=1)
        elif activation == 'sigmoid':
            return F.sigmoid(x)

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.hidden[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.hidden[-1](x)
        return tuple(self.activate(self.task_nets[task_model_id](x), self.task_list[task_model_id].output_module)
                     for task_model_id in range(len(self.task_list)))
