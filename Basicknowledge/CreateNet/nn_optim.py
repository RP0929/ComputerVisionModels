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
optim = torch.optim.SGD(mrp.parameters(),lr=0.01)
mrp_net = mrp()
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = mrp_net(imgs)
        result_loss = loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)