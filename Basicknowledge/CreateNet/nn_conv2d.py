import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("../../data",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,batch_size = 64)

class mrp(nn.Module):
    def __init__(self):
        super(mrp, self).__init__()
        self.conv1 = nn.Conv2d(3,6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        output = self.conv1(x)
        return output

mrp_net = mrp()
# print(mrp_net)

for data in dataloader:
    imgs,targets = data
    output = mrp_net(imgs)
    print(output.shape)