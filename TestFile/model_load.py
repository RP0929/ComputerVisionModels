import torch
import torchvision.models
from models.VGG.vgg import VGG16, VGG19


checkpoint = torch.load('../checkpoint/ckpt_vgg19.pth')
model = checkpoint['net']
acc = checkpoint['acc']
epoch = checkpoint['epoch']
print("acc:",acc,"epoch:",epoch)
# model = torch.load('./checkpoint/ckpt1.pth')
print(model)