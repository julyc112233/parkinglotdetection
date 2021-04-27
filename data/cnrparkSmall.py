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
str=sys.argv[0].split('/')
tmp="/"
for path in str[:-1]:
    tmp=osp.join(tmp,path)
data_targ_root=osp.join(tmp,"data","cnr_park_small")

cameras=["A","B"]



classes=["free","busy"]
# targ_root="/Users/julyc/PycharmProjects/vgg16/data/train/"

def getdataset(root):
    class_to_ind = dict(zip(classes, range(len(classes))))
    file_root=list()
    label=list()
    for camera in cameras:
        for state in classes:
            dir_root = osp.join(root, camera, state)
            for file in os.listdir(dir_root):
                if not file.lower().endswith(".jpg"):
                    continue
                file_root.append(osp.join(dir_root, file))
                label.append([class_to_ind[state]])
    return file_root,label

def dataset_split(dataset,labels):
    train_x,test_x,train_y,test_y=train_test_split(dataset,labels,test_size=0.3,random_state=0)
    for file_root,label in zip(train_x,train_y):
        # print(file_root,label)
        # exit()
        targ_root=osp.join(data_targ_root,"train")
        str=file_root.split('/')
        tmp_dir=osp.join(targ_root,str[-3],str[-2])
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        targ_dir=osp.join(tmp_dir,str[-1])
        # print(targ_dir)
        # exit()

        im=cv2.imread(file_root)
        # print(im.shape)
        im=cv2.resize(im,(224,224),interpolation=cv2.INTER_LINEAR)
        # print(im.shape)
        # exit()
        # print(targ_dir)
        cv2.imwrite(targ_dir,im)
        im2=cv2.imread(targ_dir)
        # print(im2.shape)
        # exit()

        # shutil.copyfile(file_root,targ_dir)

    for file_root,label in zip(test_x,test_y):
        targ_root=osp.join(data_targ_root,"test")
        str=file_root.split('/')
        tmp_dir=osp.join(targ_root,str[-3],str[-2])
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        targ_dir=osp.join(tmp_dir,str[-1])

        im = cv2.imread(file_root)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(targ_dir, im)


        # shutil.copyfile(file_root,targ_dir)


# print(data)
class cnrpark_small(data.Dataset):
    def __init__(self,root,transform=None):
        self.class_to_ind=dict(zip(classes,range(len(classes))))
        self.root=root
        self.file_root=list()
        self.label=list()
        self.transform=transform
        for camera in cameras:
            for state in classes:
                dir_root=osp.join(root,camera,state)
                for file in os.listdir(dir_root):
                    if not file.lower().endswith(".jpg"):
                        continue
                    self.file_root.append(osp.join(dir_root,file))
                    self.label.append([self.class_to_ind[state]])
        self.len=len(self.label)

    def __getitem__(self,index):
        im=cv2.imread(self.file_root[index])
        b, g, r = cv2.split(im)
        im = cv2.merge([r, g, b])
        # print("test:",im.shape)
        # im=transforms.RandomCrop(224, pad_if_needed=True, fill=0, padding_mode='constant')(im)
        gt=self.label[index]
        # print(im.shape)
        if self.transform is not None:
            # print(type(im))
            im=self.transform(im)
            # b, g, r = cv2.split(im)
            # print(im.shape)#[3, 224, 224]
            # im = im[:, :, :]
            # im=cv2.merge([r,g,b])
            # print(im.shape)#[224, 3, 224]
            # print(im.shape)
        # im=im.permute(2,0,1)
        # print(im.shape)
        # print(im.shape)
        # exit()
        return im,gt

    def __len__(self):
        return self.len


def test(root):
    file_root=list()
    label=list()
    class_to_ind=dict(zip(classes, range(len(classes))))
    height=[]
    weight=[]
    for camera in cameras:
        for state in classes:
            dir_root = osp.join(root, camera, state)
            for file in os.listdir(dir_root):
                if not file.lower().endswith(".jpg"):
                    continue
                file=osp.join(dir_root,file)
                # print(file)
                im=cv2.imread(file)
                height.append(im.shape[0])
                weight.append(im.shape[1])
    print(np.min(height),np.min(weight))
    # len = len(label)

# dataset,label=getdataset(cnr_small_root)
# dataset_split(dataset,label)
# test(targ_root)
