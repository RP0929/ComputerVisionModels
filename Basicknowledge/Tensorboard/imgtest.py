import torch
import torchvision.transforms
from PIL import Image
from torchvision import transforms
from models.VGG.vgg import VGG16
image_path = "./imgs/Dog.jpg"
image = Image.open(image_path)
print(image)
tranform = torchvision.transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
vgg16 = VGG16()
image= tranform(image)
checkpoint = torch.load("../../checkpoint/ckpt_vgg16.pth")
vgg16.load_state_dict(checkpoint['net'])
print(vgg16)