import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../../data",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,batch_size = 1)

class mrp(nn.Module):
    def __init__(self):
        super(mrp, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forwaed(self,x):
        out = self.model1(x)
        return out

loss = nn.CrossEntropyLoss()

mrp_net = mrp()
for data in dataloader:
    imgs, targets = data
    outputs = mrp_net(imgs)
    result_loss = loss(outputs,targets)
    result_loss.backward()
