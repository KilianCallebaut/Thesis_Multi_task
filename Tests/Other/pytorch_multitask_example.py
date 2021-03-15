from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
from sklearn.metrics import f1_score
plt.ion()   # interactive mode

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# https://github.com/sugi-chan/pytorch_multitask/blob/master/pytorch%20multi-task-Copy2.ipynb

dat = pd.read_csv('Datasets\\fgo_multiclass_labels.csv')
X = dat['image_name']
y = dat[['white', 'red',
       'green', 'black', 'blue', 'purple', 'gold', 'silver', 'gender_Female',
       'gender_Male', 'region_Asia', 'region_Egypt', 'region_Europe',
       'region_Middle East', 'fighting_type_magic', 'fighting_type_melee',
       'fighting_type_ranged', 'alignment_CE', 'alignment_CG', 'alignment_CN',
       'alignment_LE', 'alignment_LG', 'alignment_LN', 'alignment_NE',
       'alignment_NG', 'alignment_TN']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

X_train = X_train.values.tolist()
X_test = X_test.values.tolist()

#train
colors = ['white', 'red',
       'green', 'black', 'blue', 'purple', 'gold', 'silver']
color_train = y_train[colors]
color_nodes = color_train.shape[1]
color_train = color_train.values.tolist()

gender = ['gender_Female',
       'gender_Male']
gender_train = y_train[gender]
gender_nodes = gender_train.shape[1]
gender_train = gender_train.values.tolist()

region = ['region_Asia', 'region_Egypt', 'region_Europe',
       'region_Middle East']
region_train = y_train[region]
region_nodes = region_train.shape[1]
region_train = region_train.values.tolist()

fighting_style = ['fighting_type_magic', 'fighting_type_melee',
       'fighting_type_ranged']
fighting_style_train = y_train[fighting_style]
fighting_nodes = fighting_style_train.shape[1]
fighting_style_train = fighting_style_train.values.tolist()

alignment = ['alignment_CE', 'alignment_CG', 'alignment_CN',
       'alignment_LE', 'alignment_LG', 'alignment_LN', 'alignment_NE',
       'alignment_NG', 'alignment_TN']
alignment_train = y_train[alignment]
alignment_nodes = alignment_train.shape[1]
alignment_train = alignment_train.values.tolist()

print(color_nodes,gender_nodes,region_nodes,fighting_nodes,alignment_nodes)
#test
colors = ['white', 'red',
       'green', 'black', 'blue', 'purple', 'gold', 'silver']
color_test = y_test[colors]
color_nodes = color_test.shape[1]
color_test = color_test.values.tolist()

gender = ['gender_Female',
       'gender_Male']
gender_test = y_test[gender]
gender_nodes = gender_test.shape[1]
gender_test = gender_test.values.tolist()

region = ['region_Asia', 'region_Egypt', 'region_Europe',
       'region_Middle East']
region_test = y_test[region]
region_nodes = region_test.shape[1]
region_test = region_test.values.tolist()

fighting_style = ['fighting_type_magic', 'fighting_type_melee',
       'fighting_type_ranged']
fighting_style_test = y_test[fighting_style]
fighting_nodes = fighting_style_test.shape[1]
fighting_style_test = fighting_style_test.values.tolist()

alignment = ['alignment_CE', 'alignment_CG', 'alignment_CN',
       'alignment_LE', 'alignment_LG', 'alignment_LN', 'alignment_NE',
       'alignment_NG', 'alignment_TN']
alignment_test = y_test[alignment]
alignment_nodes = alignment_test.shape[1]
alignment_test = alignment_test.values.tolist()

print(color_nodes,gender_nodes,region_nodes,fighting_nodes,alignment_nodes)


class fgo_dataset(Dataset):
    def __init__(self, king_of_lists, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.king_of_lists = king_of_lists
        self.transform = transform

    def __getitem__(self, index):
        # random_index = randrange(len(self.king_of_lists[index]))

        img1 = Image.open('Datasets\\images/' + self.king_of_lists[0][index])
        img1 = img1.convert('RGB')

        gender = self.king_of_lists[1][index]  # gender
        region = self.king_of_lists[2][index]  # region
        fight = self.king_of_lists[3][index]  # fighting
        alignment = self.king_of_lists[4][index]  # alignment
        color = self.king_of_lists[5][index]  # color
        if self.transform is not None:
            img1 = self.transform(img1)
        list_of_labels = [torch.from_numpy(np.array(gender)),
                          torch.from_numpy(np.array(region)),
                          torch.from_numpy(np.array(fight)),
                          torch.from_numpy(np.array(alignment)),
                          torch.from_numpy(np.array(color))]
        # list_of_labels = torch.FloatTensor(list_of_labels)

        # print(img1.shape,len(list_of_labels),
        #     len(gender),
        #     len(region),
        #     len(fight),
        #     len(alignment),
        #    len(color))

        # torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        # print(list_of_labels)
        return img1, list_of_labels[0], list_of_labels[1], list_of_labels[2], list_of_labels[3], list_of_labels[4]

    def __len__(self):
        return len(self.king_of_lists[0])

batch_size = 16
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
train_lists = [X_train, gender_train, region_train, fighting_style_train, alignment_train, color_train]
test_lists = [X_test, gender_test, region_test, fighting_style_test, alignment_test, color_test]

training_dataset = fgo_dataset(king_of_lists = train_lists,
                               transform = data_transforms['train'] )

test_dataset = fgo_dataset(king_of_lists = test_lists,
                           transform = data_transforms['val'] )

dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                   }
dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 100


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_loss0 = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            running_loss3 = 0.0
            running_loss4 = 0.0

            running_corrects = 0
            gender_corrects = 0.0
            region_corrects = 0.0
            fighting_corrects = 0.0
            alignment_corrects = 0.0
            color_corrects = []
            total_colors = []
            # Iterate over data.
            for inputs, gen, reg, fight, ali, color in dataloaders_dict[phase]:

                inputs = inputs.to(device)

                gen = gen.to(device)
                reg = reg.to(device)
                fight = fight.to(device)
                ali = ali.to(device)
                color = color.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(inputs)
                    outputs = model(inputs)

                    # print('')
                    # print(outputs[0])
                    # print('')
                    # _, targets = y1.max(dim=0)
                    # nn.CrossEntropyLoss()(out, Variable(targets))
                    # _, preds0 = torch.max(outputs[0], 1)
                    # _, preds1 = torch.max(outputs[1], 1)
                    # _, preds2 = torch.max(outputs[2], 1)
                    # _, preds3 = torch.max(outputs[3], 1)
                    # print('preds')
                    # print(outputs[0],outputs[0].max(1))
                    # print(torch.max(outputs[0], 1),outputs[0],'potato',gen)
                    # print(outputs[0], torch.max(gen.float(), 1)[1])
                    # print(outputs[4].cpu().detach().numpy())
                    loss0 = criterion[0](outputs[0], torch.max(gen.float(), 1)[1])
                    loss1 = criterion[1](outputs[1], torch.max(reg.float(), 1)[1])
                    loss2 = criterion[2](outputs[2], torch.max(fight.float(), 1)[1])
                    loss3 = criterion[3](outputs[3], torch.max(ali.float(), 1)[1])
                    loss4 = criterion[4](outputs[4], color.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = loss0 + loss1 + loss2 + loss3 + loss4
                        # print(loss, loss0,loss1, loss2, loss3,loss4)
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss0 += loss0.item() * inputs.size(0)
                running_loss1 += loss1.item() * inputs.size(0)
                running_loss2 += loss2.item() * inputs.size(0)
                running_loss3 += loss3.item() * inputs.size(0)
                running_loss4 += loss4.item() * inputs.size(0)

                # print(torch.max(outputs[0], 1)[1],torch.max(gen, 1)[1],torch.max(outputs[0], 1)[1]==torch.max(gen, 1)[1])
                gender_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(gen, 1)[1])
                region_corrects += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(reg, 1)[1])
                fighting_corrects += torch.sum(torch.max(outputs[2], 1)[1] == torch.max(fight, 1)[1])
                alignment_corrects += torch.sum(torch.max(outputs[3], 1)[1] == torch.max(ali, 1)[1])


                color_corrects.append(
                    float(
                        sum(1.0 for a, b in zip(np.rint(outputs[4].cpu().detach().numpy()), color.float())
                            if a == b)
                        # (np.rint(outputs[4].cpu().detach().numpy()) == color.float()).sum()
                    )
                )
                total_colors.append(float((color.size()[0] * color.size(1))))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss0 = running_loss0 / dataset_sizes[phase]
            epoch_loss1 = running_loss1 / dataset_sizes[phase]
            epoch_loss2 = running_loss2 / dataset_sizes[phase]
            epoch_loss3 = running_loss3 / dataset_sizes[phase]
            epoch_loss4 = running_loss4 / dataset_sizes[phase]

            gender_acc = gender_corrects.double() / dataset_sizes[phase]
            region_acc = region_corrects.double() / dataset_sizes[phase]
            fighting_acc = fighting_corrects.double() / dataset_sizes[phase]
            alignment_acc = alignment_corrects.double() / dataset_sizes[phase]
            color_acc = float(sum(color_corrects)) / sum(total_colors)
            # print(outputs[4].cpu().detach().numpy(),color.float())

            # color_rounded_array = np.rint(outputs[4].cpu().detach().numpy())
            # color_acc_f1 = f1_score(color.float(),color_rounded_array,average='samples')
            # color_acc = color_corrects.double() / dataset_sizes[phase]
            # print()
            # print('{} Loss: {:.4f}'.format(phase, epoch_loss))#, gender_acc,region_acc,fighting_acc,alignment_acc,color_acc))
            # print('gender_acc: {:.4f}'.format(gender_acc))
            print(
                '{} total loss: {:.4f} gender loss: {:.4f} region loss: {:.4f} fighting loss: {:.4f} ali loss {:.4f} color loss {:.4f}'.format(
                    phase, epoch_loss, epoch_loss0,
                    epoch_loss1, epoch_loss2,
                    epoch_loss3, epoch_loss4))
            print('{} gender_Acc: {:.4f} '
                  'region_acc: {:.4f}  fighting_acc: {:.4f}  alignment_acc: {:.4f}  color_acc: {:.4f} '.format(
                phase, gender_acc, region_acc, fighting_acc, alignment_acc, color_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_acc:
                print('saving with loss of {}'.format(epoch_loss),
                      'improved over previous {}'.format(best_acc))
                best_acc = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet50(pretrained=True)
# for param in model_ft.parameters():
#    param.requires_grad = False
print(model_ft)
# num_ftrs = model_ft.classifier[6].in_features
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 512)


class multi_output_model(torch.nn.Module):
    def __init__(self, model_core, dd):
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
        self.y1o = nn.Linear(256, gender_nodes)
        nn.init.xavier_normal_(self.y1o.weight)  #
        self.y2o = nn.Linear(256, region_nodes)
        nn.init.xavier_normal_(self.y2o.weight)
        self.y3o = nn.Linear(256, fighting_nodes)
        nn.init.xavier_normal_(self.y3o.weight)
        self.y4o = nn.Linear(256, alignment_nodes)
        nn.init.xavier_normal_(self.y4o.weight)
        self.y5o = nn.Linear(256, color_nodes)
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

        return y1o, y2o, y3o, y4o, y5o


dd = .1
model_1 = multi_output_model(model_ft, dd)
model_1 = model_1.to(device)
print(model_1)
print(model_1.parameters())
criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.BCELoss()]
# criterion = [nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.MultiLabelSoftMarginLoss()]

# Observe that all parameters are being optimized

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
lrlast = .001
lrmain = .0001

optim1 = optim.Adam(
    [
        {"params": model_1.resnet_model.parameters(), "lr": lrmain},
        {"params": model_1.x1.parameters(), "lr": lrlast},
        {"params": model_1.x2.parameters(), "lr": lrlast},
        {"params": model_1.y1o.parameters(), "lr": lrlast},
        {"params": model_1.y2o.parameters(), "lr": lrlast},
        {"params": model_1.y3o.parameters(), "lr": lrlast},
        {"params": model_1.y4o.parameters(), "lr": lrlast},
        {"params": model_1.y5o.parameters(), "lr": lrlast},

    ])

# optim1 = optim.Adam(model_1.parameters(), lr=0.0001)#,momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim1

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


model_ft1 = train_model(model_1, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)
