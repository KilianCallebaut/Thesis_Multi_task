import types

import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Task:

    def __init__(
            self,
            name: str,
            output_labels,
            loss_function=None
            # output_module='softmax' #'sigmoid'

    ):
        super().__init__()
        self.output_labels = output_labels
        self.name = name
        self.classification_type = ''
        self.loss_function = loss_function

    def decision_making(self, output):
        # if self.classification_type == 'multi-class':
        #     return default_max_multi_class(output)
        # elif self.classification_type == 'multi-label':
        #     return default_max_multi_label(output)
        pass

    def translate_labels(self, output):
        # if self.classification_type == 'multi-class':
        #     return torch.max(output, 1)[1]
        # else:
        #     return output
        pass


class MultiClassTask(Task):

    def __init__(self,
                 name,
                 output_labels,
                 loss_function=None):
        super().__init__(name, output_labels, loss_function)
        self.classification_type = 'multi-class'
        if not loss_function:
            self.loss_function = nn.CrossEntropyLoss().to(device)
        else:
            self.loss_function = loss_function

    def decision_making(self, output):
        if len(output) == 0:
            return output
        return torch.max(output, 1)[1]

    def translate_labels(self, output):
        if len(output) == 0:
            return output
        return torch.max(output, 1)[1]


class MultiLabelTask(Task):

    def __init__(self,
                 name,
                 output_labels,
                 loss_function=None):
        super().__init__(name, output_labels, loss_function)
        self.classification_type = 'multi-label'
        if not loss_function:
            self.loss_function = nn.BCELoss().to(device)
        else:
            self.loss_function = loss_function

    def decision_making(self, output):
        if len(output) == 0:
            return output
        return (output >= 0.5).float()

    def translate_labels(self, output):
        if len(output) == 0:
            return output
        return output.float()
