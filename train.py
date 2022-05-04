import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils import data
from torchvision import transforms
import torchvision
import torchvision.models as models
import numpy as np
os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_port 29501 train.py
#  fuser -v /dev/nvidia*
def load_data_CIFAR10(batch_size, resize=None,normalize = False):
      
    train_trans = [transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor()]
    test_trans = [transforms.ToTensor()]
    if normalize:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        train_trans.insert(-1,norm)
        test_trans.insert(-1,norm)
    if resize:
        train_trans.insert(0, transforms.Resize(resize))
        test_trans.insert(0, transforms.Resize(resize))

    train_trans = transforms.Compose(train_trans)
    test_trans = transforms.Compose(test_trans)
    cifar10_train = torchvision.datasets.CIFAR10(
        root="./dataset", train=True, transform=train_trans, download=True)
    cifar10_test = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, transform=test_trans, download=True)
    sampler = DistributedSampler(cifar10_train)
    return (data.DataLoader(cifar10_train, batch_size,sampler=sampler),
            data.DataLoader(cifar10_test, batch_size))
  
seed = 50
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

resnet50 = models.resnet50(pretrained=True,progress=True)
inchannel = resnet50.fc.in_features
resnet50.fc = nn.Linear(inchannel, 10)


local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
device = torch.device('cuda', local_rank)



class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def accuracy(y_hat, y): 
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, test_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    net.eval()  # 设置为评估模式
    test_count=0
    if not device:
        device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y =y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
            test_count += 1
            if local_rank ==0:
                print(f'test_bitch={test_count}/79,test_acc={metric[0] / metric[1]}')
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, optimizer,epoch):
    count =0
    net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
    # 计算梯度并更新参数
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.mean().backward()
        optimizer.step()
        with torch.no_grad():
            count +=1
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
            print(f'epoch = {epoch},bitch = {count} /196, local_rank = {local_rank},train_acc={metric[1] / metric[2]}')
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, optimizer):  
    timer = Timer()
    for epoch in range(num_epochs):
        # train
        timer.start()
        train_metrics = train_epoch(net, train_iter, loss, optimizer,epoch)
        timer.stop()
        #accuray
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        train_loss, train_acc = train_metrics
        if local_rank ==0:
            print(f'eopch 1 ~ eopch {epoch+1} cost: {timer.sum():.1f} sec')
            print(f'train_acc={train_acc},train_loss={train_loss},test_acc={test_acc}')



def run():
    dist.init_process_group(backend="nccl")
    
    batch_size = 128
    train_iter, test_iter = load_data_CIFAR10(batch_size,resize=(224, 224))
    
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")

    
    model = resnet50.cuda(local_rank)
    ddp_model = DDP(model, [local_rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001,momentum=0.9)
    
    num_epochs= 5
    train(ddp_model,train_iter, test_iter, loss_fn, num_epochs, optimizer)
    if local_rank ==0:
        torch.save(model.state_dict(), 'resnet50_1.params')
        torch.save(model, 'net.pt')
        print('param saved')
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) well done")
 