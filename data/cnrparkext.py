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

def cnrext_dataset_split(dataset_root,str="train"):
    if str=="train":
        txt_root=osp.join(dataset_root,"LABELS/train.txt")
    else:
        txt_root=osp.join(dataset_root,"LABELS/test.txt")
    f=open(txt_root,"r",encoding='utf-8')
    txt_data=[]
    label=[]
    line=f.readline().strip()
    while line:
        txt_data.append(line.split(' '))
        line=f.readline().strip()
    targ_root=osp.join(data_targ_root,str)
    # print(targ_root)
    # exit()
    for img,label in txt_data:
        targ_dir=osp.join(targ_root,img)
        tmp_dir="/"
        tmp=targ_dir.split('/')[:-1]
        # print(tmp)
        for a in tmp:
            tmp_dir=osp.join(tmp_dir,a)
        file_root=osp.join(dataset_root,"PATCHES",img)

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        im=cv2.imread(file_root)
        im=cv2.resize(im,(224,224),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(targ_dir,im)

class cnrext(data.Dataset):
    def __init__(self,root,transform=None,str=None):
        self.class_to_ind=dict(zip(classes,range(len(classes))))
        self.root=root
        self.file_root=list()
        self.transform=transform
        if str=="train":
            self.txt_root=osp.join(root,"train.txt")
        else:
            self.txt_root=osp.join(root,"test.txt")
        self.label=list()
        f=open(self.txt_root,"r",encoding='utf-8')
        line=f.readline().strip()
        txt_data=[]
        targ_root = osp.join(data_targ_root, str)
        while line:
            # txt_data.append(line.split(' '))
            a,b=line.split(" ")
            self.file_root.append(targ_root+'/'+a)
            self.label.append([self.class_to_ind[int(b)]])
            line = f.readline().strip()

        self.len=len(self.label)

    def __getitem__(self,index):

        im=cv2.imread(self.file_root[index])
        b, g, r = cv2.split(im)
        im = cv2.merge([r, g, b])

        gt=self.label[index]
        # print(gt)
        if self.transform is not None:

            boxes=None
            im,boxes,gt=self.transform(im,boxes,gt)
            im=im[:,:,(2,1,0)]

        return torch.from_numpy(im).permute(2, 0, 1),gt

    def __len__(self):
        return self.len

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        return cv2.imread(self.file_root[index], cv2.IMREAD_COLOR)
    def pull_label(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        return self.label[index]