import torchvision
import torch

def vgg16(pretrain=None):
    model=torchvision.models.vgg16(pretrain)
    # 冻结参数
    # for parma in model.parameters():
    #     parma.requires_grad = False
    # print(model.classifier)
    model.classifier=torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))
    return model
    # return model





