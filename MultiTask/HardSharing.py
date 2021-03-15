from torch import nn
from torch.autograd.grad_mode import F

from MultiTask.FFNN import FFNN
from MultiTask.TaskDependentLayers import TaskDependentLayers
from MultiTask.TaskDependentSoftmax import TaskDependentSoftmax


# https://gist.github.com/AvivNavon/cf2071ffaadfb11dded915f7f4bd638e
class HardSharing(nn.Module):
    """FFNN with hard parameter sharing
    """

    # Hyperparameters for our network
    # Input: 7 (summaries) x # coefficients (typically 24 for mel-scaled filterbanks)
    # 1 hidden layer with 128 units
    # ReLU activation
    # 1 hidden layer with 128 units
    # Another ReLU
    # 1 hidden layer with 128 units
    # Another ReLU
    # Output layer with softmax (task dependent)

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
            task_list,
            dropout_rate=.1,
    ):
        super().__init__()

        self.model = nn.Sequential()

        self.model.add_module(
            'hard_sharing',
            FFNN(
                input_size=input_size,
                hidden_size=hidden_size,
                n_hidden=n_hidden-1,
                n_outputs=hidden_size,
            )
        )

        self.model.add_module(
            'task_specific',
            TaskDependentLayers(
                hidden_size=hidden_size,
                task_list=task_list
            )
        )

    def forward(self, x):
        return self.model(x)
