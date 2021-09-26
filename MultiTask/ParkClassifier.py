from typing import List

from torch import nn

from Tasks.Task import Task


class ParkClassifier(nn.Module):
    def __init__(self, task_list: List[Task]):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, len(t.output_labels)),
                nn.LogSoftmax(dim=-1) if t.classification_type == 'multi-class' else nn.Sigmoid()
            )
            for t in task_list]
        )

    def forward(self, x):
        x = x.squeeze()
        x = self.classifier.forward(x)
        return tuple(head.forward(x) for head in self.output_heads)
