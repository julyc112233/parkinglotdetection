import argparse
import sys
import time

from ssdaugumentations import SSDAugmentation
from data import *
import torchvision
import torch
import os.path as osp
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data.cnrparkSmall import *
from data import *
from net import *
from data.cnrparkext import *
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
im_root="./data/example.jpg"
net=malexnet()

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x
class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
transform=BaseTransform(224,(104, 117, 123))
im=cv2.imread("./data/example.jpg", cv2.IMREAD_COLOR)
# x=im.astype(np.float32)
# print(im.shape)
im=im.astype(np.float32)
# print(im.shape)
x = torch.from_numpy(transform(im)[0]).permute(2, 0, 1)
x = Variable(x.unsqueeze(0))
# print(x.shape)
y=net(x)
print(y.shape)


