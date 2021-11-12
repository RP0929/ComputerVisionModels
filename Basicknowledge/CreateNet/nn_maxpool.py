import torch
from torch import nn

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)
input = torch.reshape(input,(-1,1,5,5))

class mrp(nn.Module):
    def __init__(self):
        super(mrp, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,x):
        output = self.maxpool1(x)
        return output

mrp_net = mrp()
output = mrp_net(input)

print(output)
