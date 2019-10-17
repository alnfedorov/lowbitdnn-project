import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qtorch.nn.Int8Conv2d import Int8Conv2d
from qtorch.nn.functional import qmax_pool2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN, STD = -95, 78.5

class ToInt8Tensor(object):
    def __call__(self, pic):
        ten = (torch.tensor(np.asarray(pic)).to(torch.float32) - 128)
        ten = (ten - MEAN) / STD
        ten.clamp_(-3, 3)

        qmin, qmax = -60, 60
        fmin, fmax = -3, 3
        inv_scale = (qmax - qmin) / (fmax - fmin)
        ten.mul_(inv_scale).round_()
        ten.clamp_(qmin, qmax)
        return ten.to(torch.int8).reshape(1, 28, 28)

    def __repr__(self):
        return self.__class__.__name__ + '()'


BATCH_SIZE = 60000

class QCNNClassifier(nn.Module):
    def __init__(self):
        super(QCNNClassifier, self).__init__()
        self.conv1 = Int8Conv2d(4,  4,  (3, 3), (1, 1), (0, 0), (1, 1))
        self.conv2 = Int8Conv2d(4,  4, (3, 3), (1, 1), (0, 0), (1, 1))
        self.conv3 = Int8Conv2d(4, 4, (3, 3), (1, 1), (0, 0), (1, 1))
        self.conv4 = Int8Conv2d(4, 12, (1, 1), (1, 1), (0, 0), (1, 1))

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28, 1)
        x = F.pad(x, [0, 3]).contiguous()

        scale, x = self.conv1(0.05, x)
        x = qmax_pool2d(x, (2, 2), (2, 2), (0, 0))
        x = F.relu(x, inplace=True)

        scale, x = self.conv2(scale, x)
        x = qmax_pool2d(x, (2, 2), (2, 2), (0, 0))
        x = F.relu(x, inplace=True)

        scale, x = self.conv3(scale, x)
        x = qmax_pool2d(x, (3, 3), (1, 1), (0, 0))
        x = F.relu(x, inplace=True)

        scale, x = self.conv4(scale, x)
        x = F.relu(x, inplace=True)
        assert x.shape[2] == x.shape[3] == 1
        x = x.reshape(-1, 12)[:, :-2]
        return x

# class FCNNClassifier(nn.Module):
#     def __init__(self):
#         super(FCNNClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1,  4,  (3, 3), (1, 1), (0, 0), bias=False)
#         self.bn1 = nn.BatchNorm2d(4, track_running_stats=False)
#
#         self.conv2 = nn.Conv2d(4,  4, (3, 3), (1, 1), (0, 0), bias=False)
#         self.bn2 = nn.BatchNorm2d(4, track_running_stats=False)
#
#         self.conv3 = nn.Conv2d(4, 4, (3, 3), (1, 1), (0, 0), bias=False)
#         self.bn3 = nn.BatchNorm2d(4, track_running_stats=False)
#
#         self.conv4 = nn.Conv2d(4, 10, (1, 1), (1, 1), (0, 0), bias=False)
#         for p in self.parameters():
#             p.requires_grad = False
#
#     def forward(self, x):
#         x = x.reshape(-1, 1, 28, 28)
#
#         x = self.conv1(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), padding=0, return_indices=False)
#         x = F.relu(x, inplace=True)
#         x = self.bn1(x)
#
#         x = self.conv2(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), padding=0, return_indices=False)
#         x = F.relu(x, inplace=True)
#         x = self.bn2(x)
#
#         x = self.conv3(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), padding=0, return_indices=False)
#         x = F.relu(x, inplace=True)
#         x = self.bn3(x)
#
#         x = self.conv4(x)
#         x = F.max_pool2d(x, kernel_size=2, stride=(2, 2), padding=0, return_indices=False)
#         x = F.relu(x, inplace=True)
#         assert x.shape[2] == x.shape[3] == 1
#         x = x.reshape(-1, 10)
#         return x

# Computing device
device = torch.device("cuda")
model = QCNNClassifier().to(device)
# model = FCNNClassifier().to(device)

# Data loaders
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                          download=True,
                                                          train=True,
                                                          transform=transforms.Compose([ToInt8Tensor()])),
                                           batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=4)
batches = []
for b in train_loader:
    batches.append([x.to(device, non_blocking=True) for x in b])
# download and transform test dataset
# test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
#                                                          download=True,
#                                                          train=False,
#                                                          transform=transforms.Compose([ToInt8Tensor()])),
#                                           batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=4)

def objective():
    coincide = torch.tensor(0, dtype=torch.int64, device=device)
    # loss = 0.
    with torch.no_grad():
        for batch in batches:
            image, target = batch
            predict = model(image)

            coincide += torch.sum(predict.argmax(dim=-1) == target)
            # predict = F.log_softmax(predict)
            # loss += float(F.nll_loss(predict, target, reduction='mean').to('cpu'))
    coincide = float(coincide.to('cpu')) / (len(train_loader) * BATCH_SIZE)
    # print(f"Accuracy is {coincide}")
    return (1 - coincide) * 10 # energy
    # loss = loss / len(batches)
    # return loss * 1000

from optimize.simulated_annealing import simulated_annealing, linear_schedule, fast_annealing, \
    metropolis_acceptance_prob
from functools import partial

q_v, q_a = 2.62, -5
annealing = partial(fast_annealing, lower=-5, upper=5)#partial(generalized_annealing, q_v=q_v, lower=-5, upper=5)#
temperature = linear_schedule#partial(gsa_schedule, q_v=q_v)
acceptance = metropolis_acceptance_prob#partial(gsa_acceptance_prob, q_a=q_a)

history = simulated_annealing(list(model.parameters()), objective, annealing, temperature, acceptance,
                              initial_temp=0.25, max_eval=10e10, restart_temp_ratio=0.02)
for x, loss in history.history:
    print(loss)
