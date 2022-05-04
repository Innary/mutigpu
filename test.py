import torch
from PIL import Image
import PIL
from torchvision import transforms
import torchvision
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import random

# test_trans = [ transforms.ToTensor(),transforms.Resize((224,224))]
# test_trans = transforms.Compose(test_trans)
test_trans = [transforms.Resize((224,224))]

test_trans=transforms.Compose(test_trans)
cifar10_test = torchvision.datasets.CIFAR10(root="./dataset", train=False,transform=test_trans, download=True)
net = torch.load('net.pt').to(torch.device('cuda:0'))

def get__labels(label):  
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return text_labels[int(label)]

for _ in range(5):
    num = random.randint(0,9999)

    img =cifar10_test[num][0]
    y=cifar10_test[num][1]
    print(f"====+=====cifar10 test image {_+1}=====+===")
    print("|                                      |")
    print(f"| No.{num} image's ground truth === "+ get__labels(y)+' |')
    print("|                                      |")
    print('=======================================')
    # img.save('output.jpg')
    # img=np.array(Image.open('output.jpg').resize((224,224)), dtype=np.float32)
    img=np.array(img.resize((224,224)), dtype=np.float32)
    img=img/255
    img=img.transpose(2,0,1)
    img=img[np.newaxis,...] 
    img=torch.from_numpy(img).to(torch.device('cuda:0'))
    net.eval()
    with torch.no_grad():
        
        print("|                                      |")
        print(f"| No.{num} image's output     === {get__labels(torch.nn.Softmax(dim=1)((net(img))).argmax(axis=1)[0])} |")
        print("|                                      |")
        print('=======================================')