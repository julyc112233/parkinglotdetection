import os.path as osp
import os
import torch.utils.data as data
import cv2
import torch
import shutil
import numpy as np
import sys
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

classes=[0,1]

weather=["OVERCAST","SUNNY","RAINY"]
dataset_root=""
str=os.path.abspath(sys.argv[0]).split("/")
tmp="/"
for path in str[:-1]:
    tmp=osp.join(tmp,path)
data_targ_root=osp.join(tmp,"data","cnrext")

class cnrext_dataset_split(dataset_root,str="train"):
    txt_root=osp.join(dataset_root,"LABELS/train.txt")
    # print(txt_root)
    # exit()
    f=open(dataset_root,"r",encoding='utf-8')
    txt_data=[]
    label=[]
    line=f.readline()
    while line:
        txt_data.append(eval(line))
        line=f.readline()
    targ_root=osp.join(data_targ_root,str)
    for img,label in txt_data:
        targ_dir=osp.join(targ_root,img)
        tmp_dir="/"
        for a in targ_dir.split('/')[:-1]:
            tmp_dir=osp.join(tmp,a)
        file_root=osp.join(dataset_root,img)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        im=cv2.imread(img)
        im=cv2.resize(im,(224,224),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(targ_dir,im)

class cnrext(data.Dataset):
    def __init__(self,root,transform=None):
        self.class_to_ind=dict(zip(classes,range(len(classes))))
        self.root=root
        self.file_root=list()
        self.transform=transform
        f=open(root,"r",encoding='utf-8')
        line=f.readline()
        txt_data=[]
        while line:
            txt_data.append(eval(line))
            line = f.readline()
        self.file_root=txt_data[0,:]
        self.label=txt_data[1,:]

        self.len=len(self.label)

    def __getitem__(self,index):
        im=cv2.imread(self.file_root[index])
        b, g, r = cv2.split(im)
        im = cv2.merge([r, g, b])

        gt=self.label[index]
        if self.transform is not None:

            boxes=None
            im,boxes,gt=self.transform(im,boxes,gt)
            im=im[:,:,(2,1,0)]

        return torch.from_numpy(im).permute(2, 0, 1),gt

    def __len__(self):
        return self.len