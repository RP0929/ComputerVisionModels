import torch
import torchvision.models

resnet18 = torchvision.models.resnet18(pretrained=False)
resnet18.load_state_dict(torch.load("./checkpoint/ckpt1.pth"))
# model = torch.load('./checkpoint/ckpt1.pth')
print(resnet18)