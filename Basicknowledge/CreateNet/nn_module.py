import torch
from torch import nn

class mrp(nn.Module):
    def __init__(self):
        super(mrp, self).__init__()

    def forward(self,x):
        output = x + 1
        return output

mrp_net = mrp()
x = torch.tensor(1.0)
output = mrp_net(x)
print(output)