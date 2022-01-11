import torch
from torch import nn
import torch.nn.functional as F

# https://github.com/sugi-chan/pytorch_multitask/blob/master/pytorch%20multi-task-Copy2.ipynb
class multi_output_model(torch.nn.Module):


    def __init__(self, model_core, dd, nodes):
        super(multi_output_model, self).__init__()

        self.resnet_model = model_core

        self.x1 = nn.Linear(512, 256)
        nn.init.xavier_normal_(self.x1.weight)

        self.bn1 = nn.BatchNorm1d(256, eps=2e-1)
        self.x2 = nn.Linear(256, 256)
        nn.init.xavier_normal_(self.x2.weight)
        self.bn2 = nn.BatchNorm1d(256, eps=2e-1)
        # self.x3 =  nn.Linear(64,32)
        # nn.init.xavier_normal_(self.x3.weight)
        # comp head 1

        # heads
        self.y1o = nn.Linear(256, nodes[0])
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(256, nodes[1])
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256, nodes[2])
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(256, nodes[3])
        nn.init.xavier_normal_(self.y4o.weight)
        self.y5o = nn.Linear(256, nodes[4])
        nn.init.xavier_normal_(self.y5o.weight)

        self.d_out = nn.Dropout(dd)

    def forward(self, x):
        x1 = self.resnet_model(x)
        # x1 =  F.relu(self.x1(x1))
        # x1 =  F.relu(self.x2(x1))

        x1 = self.bn1(F.relu(self.x1(x1)))
        x1 = self.bn2(F.relu(self.x2(x1)))
        # x = F.relu(self.x2(x))
        # x1 = F.relu(self.x3(x))

        # heads
        y1o = F.softmax(self.y1o(x1), dim=1)
        y2o = F.softmax(self.y2o(x1), dim=1)
        y3o = F.softmax(self.y3o(x1), dim=1)
        y4o = F.softmax(self.y4o(x1), dim=1)
        y5o = torch.sigmoid(self.y5o(x1))  # should be sigmoid

        # y1o = self.y1o(x1)
        # y2o = self.y2o(x1)
        # y3o = self.y3o(x1)
        # y4o = self.y4o(x1)
        # y5o = self.y5o(x1) #should be sigmoid

        return torch.cat(tuple((y1o, y2o, y3o, y4o, y5o)), dim=1)

