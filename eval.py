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
HOME = os.path.expanduser("~")
need_split_data=True
targ_root = osp.join(HOME, ".jupyter/cnrpark/parkinglotdetection/data/cnrext")
data_root = osp.join(HOME, ".jupyter/cnrpark")

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

def test_result(net,model_root,test_data,cuda):
    net.load_state_dict(torch.load(model_root))
    if cuda:
        net=net.cuda()
        cudnn.benchmark=True
    len=len(test_data)
    for i in range(test_data):
        im,label=test_data(i)
        if cuda :
            im=im.cuda()
        y=net(im)
        y=torch.nn.Softmax()
        y=torch.argmax(y)
        print(y)
        exit()

if __name__ == '__main__':
    if need_split_data:
        print("split data to train ans test...")
        # im,label=getdataset(data_root)
        # print(data_root)
        # if args.dataset_name == "cnrext":
        cnrext_dataset_split(data_root, "test")
        # else:
        #     if args.dataset_name == "cnrsmall":
        #         im, label = getdataset(data_root)
        #         cnrsmall_dataset_split(im, label)
    net=vgg16()
    model_root="weights/vgg16_cnrparkext120000.pth"
    test_data=cnrext(targ_root,base_transform(net.size(),(104, 117, 123)),str="test")
    test_result(net,model_root,test_data,cuda=False)

