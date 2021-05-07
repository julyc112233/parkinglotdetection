import torchvision
import torch
import torch.nn as nn

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
class malexnet(nn.Module):

    def __init__(self, num_classes=1000):
        super(malexnet, self).__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3,16,kernel_size=11,stride=4,padding=(2, 2)),
                              torch.nn.ReLU(inplace=True),
                              torch.nn.LocalResponseNorm(4),
                              torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                              torch.nn.Conv2d(16, 20, kernel_size=5, stride=1, padding=(2, 2)),
                              torch.nn.ReLU(inplace=True),
                              torch.nn.LocalResponseNorm(4),
                              torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
                              torch.nn.Conv2d(20, 30, kernel_size=3, stride=1, padding=(2, 2)),
                              torch.nn.ReLU(inplace=True),
                              torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.classifier = torch.nn.Sequential(torch.nn.Linear(1470,48),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(48,2))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
def malex():
    model=malexnet()
    return model
# print(torchvision.models.alexnet())
# AlexNet(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace=True)
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace=True)
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
#   (classifier): Sequential(
#     (0): Dropout(p=0.5, inplace=False)
#     (1): Linear(in_features=9216, out_features=4096, bias=True)
#     (2): ReLU(inplace=True)
#     (3): Dropout(p=0.5, inplace=False)
#     (4): Linear(in_features=4096, out_features=4096, bias=True)
#     (5): ReLU(inplace=True)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )





