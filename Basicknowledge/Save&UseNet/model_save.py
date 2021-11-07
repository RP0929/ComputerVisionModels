import torch
from torch import nn
import torchvision


vgg16 = torchvision.models.vgg16(pretrained=False)
#保存方式1
torch.save(vgg16,"vgg16_method1.pth")

#保存方式2:保存参数（官方推荐）
#在这种方式下我们把vgg16的参数保存成为参数字典的形式
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#陷阱
class mrp(nn.Module):
    def __init__(self):
        super(mrp, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x = self.conv1
        return x

mrp_net = mrp()
torch.save(mrp_net,"mrp_net.pth")