import argparse
import sys
import time


from tqdm import tqdm
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
import torch.nn.functional as F
from data.cnrparkext import *
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
HOME = os.path.expanduser("~")


need_split_data=False

targ_root = osp.join(HOME, ".jupyter/cnrpark/parkinglotdetection/data/cnrext")
data_root = osp.join(HOME, ".jupyter/cnrpark")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
# parser.add_argument('--trained_model', default=None,
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--save_folder', default=None, type=str,
#                     help='Dir to save results')
# parser.add_argument('--visual_threshold', default=None, type=float,
#                     help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
# parser.add_argument('--voc_root', default=None, help='Location of VOC root directory')
# parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# if not os.path.exists(args.save_folder):
#     os.mkdir(args.save_folder)

os.environ["CUDA_VISIBLE_DEVICES"] ="9"

cuda_device=9


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

def test_result(net,model_root,test_data,transform=None,cuda=False):
    print("loading weight...")
    net.load_state_dict(torch.load(model_root))
    print("loading finished...")
    net.eval()
    if cuda:
        net=net.cuda(cuda_device)
        cudnn.benchmark=True
    len=test_data.len
    print("start evaluate...")
    pred=0
    for i in tqdm(range(len)):
    # for i in range(len):
        im=test_data.pull_image(i)
        label=test_data.pull_label(i)
        x = torch.from_numpy(transform(im)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda :
            x=x.cuda(cuda_device)
        y=net(x)
        y=F.softmax(y,dim=1)
        y=torch.argmax(y)
        # y=y.cpu()
        # y=y.numpy()
        if y ==label[0]:
            # print("yes")
            pred+=1
    acc=pred/len
    print("Acc:",acc)


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
    net=malexnet()
    for file in os.listdir("weights"):
        if not file.endswith(".pth") :
            continue
        if file.find("malex")==-1:
            continue
        print(file)
        model_root=osp.join("weights",file)
    # model_root="weights/vgg16_cnrparkext100000.pth"
        test_data=cnrext(targ_root,str="test")
        test_result(net,model_root,test_data,transform=BaseTransform(224,(104, 117, 123)),cuda=True)

