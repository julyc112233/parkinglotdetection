import sys
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


root="/Users/julyc/PycharmProjects/vgg16/data/train/A/busy"
transform = transforms.Compose([
                                    # transforms.ToTensor(),  # 转化成张量数据结构
                                    transforms.Resize((224,224)),  # 对原始图片进行裁剪
                                    # transforms.Pad(224,fill=0,padding_mode='constant'),
                                    # transforms.ToTensor(),  # 转化成张量数据结构
                                    # transforms.RandomCrop(224, padding=None, pad_if_needed=True, fill=0, padding_mode="constant"),
                                    transforms.Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])])  # 用均值和标准差对张量图像进行归一化
# for file in os.listdir(root):
#     if not file.lower().endswith('.jpg'):
#         continue
#     str=osp.join(root,file)
#     print(str)
#     im=cv2.imread(str)
#     im=transform(im)
#     im=im.numpy().transpose((1, 2, 0))
#     print(im.shape)
#     exit()
str=sys.argv[0].split('/')
tmp=""
for path in str[:-1]:
    tmp=osp.join(tmp,path)
# str=osp.join(str[:-1])
print(tmp)

