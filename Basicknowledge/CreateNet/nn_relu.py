import torch
from torch import nn

input = torch.tensor([[1,-0.5],
                      [-1,3]])

x = torch.reshape(input,(-1,1,2,2))
print(x.size())
print(x)


class mrp(nn.Module):
    def __init__(self):
        super(mrp, self).__init__()
        #inplace = True 表示直接在x对象中改变原始值。
        #inplace = False 表示不改变x的原始值，会创建新的对象，返回的新对象是处理后的值。
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self,x):
        output = self.sigmoid1(x)
        return output

mrp_net = mrp()

output = mrp_net(x)

print(output)