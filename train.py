# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
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

from torch.autograd import Variable

# 数据分割只跑一次，分割完记得改成False
need_split_data=False



HOME = os.path.expanduser("~")
data_root=osp.join(HOME,".jupyter/cnrpark/CNRPark-Patches-150x150")
batch_size=64
num_work=32
epoch_num=50
lr=0.0001
momentum=0.9
weight_decay=5e-4


MEANS = (104, 117, 123)

targ_root=osp.join(HOME,".jupyter/cnrpark/parkinglotdetection/data/cnr_park_small/train/")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=data_root,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--cuda_device', default=3, type=int,
                    help='specific the cuda device')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# if torch.cuda.is_available():
# #     if args.cuda:
# #         torch.set_default_tensor_type('torch.cuda.FloatTensor')
# #     if not args.cuda:
# #         print("WARNING: It looks like you have a CUDA device, but aren't " +
# #               "using CUDA.\nRun with --cuda for optimal training speed.")
#       torch.set_default_tensor_type('torch.FloatTensor')
# else:
torch.set_default_tensor_type('torch.FloatTensor')
# from torch.multiprocessing import set_start_method

# try:

#     set_start_method('spawn')

# except RuntimeError:
#     pass

if args.visdom:
    import visdom
    viz = visdom.Visdom()
def net_train():
    print("loading net...")

    net = vgg16(pretrain=True)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    print("change to train mode ...")
    net.train()

    train_loss=0
    print("process_data...")
    dataset=cnrpark_small(targ_root,transform=SSDAugmentation(224,MEANS))
    epoch_size = len(dataset) // batch_size

    if args.visdom:
        vis_title = 'loss'
        vis_legend = ['Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader=torch.utils.data.DataLoader(dataset,batch_size,num_workers=num_work,shuffle=True,pin_memory=True)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    batch_iterator = iter(data_loader)
    cost=nn.CrossEntropyLoss()
    running_loss=0.0
    epoch=0
    print("start train...")

   
    for iteration in range(0,50000):
        if(iteration%epoch_size==0):
            running_loss=0
            epoch+=1
        try:
            images,targets=next(batch_iterator)
        except StopIteration:
            pass
        with torch.no_grad():
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images=Variable(images)
                targets=[Variable(ann) for ann in targets]
        t0=time.time()
        out=net(images)
        optimizer.zero_grad()

        loss=cost(out,targets[0])
        loss.backward()
        optimizer.step()
        t1=time.time()

        running_loss+=loss.data
        if iteration % 10==0:
            print('timer:%.4f sec.'%(t1-t0))
            print('iter'+repr(iteration)+'||loss:%.4f||'%(loss.item()),end=' ')
        if args.visdom:
            update_vis_plot(iteration, loss.item(),
                            iter_plot, epoch_plot, 'append')
        if iteration !=0 and iteration % 5000==0:
            print('Saving state,iter:',iteration)
            torch.save(net.state_dict,'weights/vgg16_cnrpark'+repr(iteration)+'.pth')
        # running_correct += torch.sum(out== y.data)
# net_train()
def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 1)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )
def update_vis_plot(iteration, loss, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 1)).cpu() * iteration,
        Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 1)).cpu(),
            Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )

if __name__ == '__main__':
    if need_split_data:
        print("split data to train ans test...")

        im,label=getdataset(data_root)
        dataset_split(im,label)
    net_train()

