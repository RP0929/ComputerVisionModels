import torch

#方式1->使用保存方式1，加载模型
# model = torch.load('vgg16_method1.pth')
# print(model)

#方式2，加载模型(1)
import torchvision.models
#torch.load('vgg16_method2.pth')

#方式2，加载模型(2)
# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
# print(vgg16)

#陷阱1
from torch import nn
# class mrp(nn.Module):
#     def __init__(self):
#         super(mrp, self).__init__()
#         self.conv1 = nn.Conv2d(3,64,kernel_size=3)
#
#     def forward(self,x):
#         x = self.conv1
#         return x

from model_save import *
model = torch.load('mrp_net.pth')
print(model)