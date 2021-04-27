# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import time

from data import *
import torchvision
import torch
import os.path as osp
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from data.cnrparkSmall import *
from data import *
from net import *

from torch.autograd import Variable

# 数据分割只跑一次，分割完记得改成False
need_split_data=True


HOME = os.path.expanduser("~")
data_root=osp.join(HOME,"downloads/CNRPark-Patches-150x150")
batch_size=32
num_work=12
epoch_num=50
lr=0.0001
momentum=0.9
weight_decay=5e-4

targ_root="/Users/julyc/PycharmProjects/vgg16/data/cnr_park_small/train/"




def net_train():
    print("loading net...")
    net = vgg16(pretrain=True)
    # print("test")
    print("change to train mode ...")
    net.train()

    transform = transforms.Compose([
                                    transforms.ToTensor(),  # 转化成张量数据结构
                                    # transforms.Resize((224,224)),  # 对原始图片进行裁剪
                                    # transforms.Pad(224,fill=0,padding_mode='constant'),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])  # 用均值和标准差对张量图像进行归一化
    train_loss=0
    print("process_data...")
    dataset=cnrpark_small(targ_root,transform=transform)
    epoch_size = len(dataset) // batch_size
    data_loader=torch.utils.data.DataLoader(dataset,batch_size,num_workers=num_work,shuffle=True,pin_memory=True)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    batch_iterator = iter(data_loader)
    cost=nn.CrossEntropyLoss()
    running_loss=0.0
    epoch=0
    print("start train...")
    # running_correct =0
    for iteration in range(0,5000):
        if(iteration%epoch_size==0):
            running_loss=0
            epoch+=1
        images,targets=next(batch_iterator)
        images=Variable(images)
        # print(images.shape)
        with torch.no_grad():
            targets=[Variable(ann) for ann in targets]
        # print(targets)
        t0=time.time()
        out=net(images)
        # print(out)
        # print(torch.max(out,1))
        # _,out=torch.max(out,1)
        # print(out.shape)
        # exit()
        optimizer.zero_grad()
        # print(type(out))
        # print(type(targets),targets.shape)
        # exit()
        loss=cost(out,targets[0])
        loss.backward()
        optimizer.step()
        t1=time.time()

        running_loss+=loss.data
        if iteration % 10==0:
            print('timer:%.4f sec.'%(t1-t0))
            print('iter'+repr(iteration)+'||loss:%.4f||'%(loss.item()),end=' ')
        if iteration !=0 and iteration % 5000==0:
            print('Saving state,iter:',iteration)
            torch.save(net.state_dict,'weights/vgg16_cnrpark'+repr(iteration)+'.pth')
        # running_correct += torch.sum(out== y.data)
# net_train()

if __name__ == '__main__':
    if need_split_data:
        print("split data to train ans test...")
        im,label=getdataset(data_root)
        dataset_split(im,label)
    net_train()
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
