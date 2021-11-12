from torch import nn
import torch

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

x = torch.ones((64,3,32,32))
mrp_net = mrp()
out = mrp_net(x)
print(out)