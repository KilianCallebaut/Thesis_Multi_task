import torch
from torch import nn


# from  https://gist.github.com/AvivNavon/cf2071ffaadfb11dded915f7f4bd638e#file-hard_sharing-py

class FFNN(nn.Module):
    # Hyperparameters for our network
    # Input: 7 (summaries) x # coefficients (typically 24 for mel-scaled filterbanks)
    # 1 hidden layer with 128 units
    # ReLU activation
    # 1 hidden layer with 128 units
    # Another ReLU
    # 1 hidden layer with 128 units
    # Another ReLU
    # Output layer with softmax (task dependent)

    """Simple FF network with multiple outputs.
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            n_hidden,
            n_outputs,
            dropout_rate=.1,
    ):
        """
        :param input_size: input size
        :param hidden_size: common hidden size for all layers
        :param n_hidden: number of hidden layers
        :param n_outputs: number of outputs
        :param dropout_rate: dropout rate
        """
        super().__init__()
        assert 0 <= dropout_rate < 1
        self.input_size = input_size

        h_sizes = [self.input_size] + [hidden_size for _ in range(n_hidden)] + [n_outputs]

        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(
                nn.Linear(
                    h_sizes[k],
                    h_sizes[k + 1]
                )
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)
            # x = self.dropout(x)
        return x

        # for layer in self.hidden[:-1]:
        #     x = layer(x)
        #     x = self.relu(x)
        #     # x = self.dropout(x)
        #
        # return self.hidden[-1](x)
