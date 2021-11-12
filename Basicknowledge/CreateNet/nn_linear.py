import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("../../data",train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,batch_size = 64)

class mrp(nn.Module):
    def __init__(self):
        super(mrp, self).__init__()
        self.lnear1 = nn.Linear(196600,10)

    def forward(self,x):
        output = self.conv1(x)
        return output

mrp_net = mrp()
# print(mrp_net)

for data in dataloader:
    imgs,targets = data
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = nn.Flatten(imgs)
    output = mrp_net(output)
    print(output.shape)