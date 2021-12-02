import os

import torchvision.datasets
from matplotlib.pyplot import imshow
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from models.ResNet.resnet import *
## cudnn???
import torch.backends.cudnn as cudnn


device = 'cuda'
epsilons = [0,1/256,2/256,3/256,4/256,5/256,6/256,7/256,8/256,9/256,10/256,11/256,12/256,13/256,14/256,15/256,16/256]
pretranined_model = './checkpoint/ckpt_resnet18_NoNorm_test.pth'
use_cuda = True
checkpoint = torch.load(pretranined_model)
writer = SummaryWriter('./adv/adversarial_samples_PGD_BatchSize/')
print('==>Preparing CIFAR 10 data...')
#transform

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR10 dataset and dataloader declaration
test_set = torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=False,num_workers=8)
epoch = 0
image_num = 0

print("==>Building models..")
model = resnet18()
model = model.to(device)

#Remember use DataParaller!
if device == 'cuda':
    model = torch.nn.DataParallel(model) #!!!
    cudnn.benchmark = True

model.load_state_dict(checkpoint['net'])
print(checkpoint['acc'])
#we have to use model.eval()
model.eval()

#The fgsm_attack function takes three inputs
#mage is the original clean image (x)
#epsilon is the pixel-wise perturbation amount (epsilonϵ)
#data_grad is gradient of the loss w.r.t the input image

#PGD attack
def pgd_attack(model,images,labels,eps=0.3,alpha=2/255,iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    ori_images = images.data
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs,labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images-ori_images,min=-eps,max=eps)
        images = torch.clamp(ori_images+eta,min=0,max=1).detach_()
    return images

def test(model,device,test_loader,epsilon):
    # Accuracy counter
    correct = 0
    total = 0
    global epoch,image_num
    epoch += 1
    #Loop over all examples in test set
    for batch_idx,(data,target) in enumerate(test_loader):
        #Call FGSM Attack
        perturbed_data = pgd_attack(model,data,target,alpha=epsilon)
        target = target.to(device)
        total+= target.size(0)
        if batch_idx % 30 == 0 :
            print("epsilon:",epsilon,"batch_idx",batch_idx)
        if batch_idx==9:
            print("adding ori and adv Images now!")
            writer.add_images("Origanel",data,epoch)
            writer.add_images("Adv Image",perturbed_data,epoch)

        outputs = model(perturbed_data)
        # Calculate the loss

        # Check for success
        _,final_pred = outputs.max(1, keepdim=True)
        #if final_pred.item() == target.item():
            #correct += 1
            #Special case for saving 0 epsilon examples


        correct += torch.eq(final_pred.squeeze(1),target).sum().item()
        #imshow(torchvision.utils.make_grid(perturbed_data.cpu().data,normalize=True))
    #Calculate final accuracy for this epsilon
    final_acc = 100.*correct/total
    writer.add_scalar("Attack Acc:",final_acc,epoch)
    print("Epsilon: {}\tTest Accuracy = {}/{}={}".format(epsilon,correct,total,final_acc))
    return final_acc


#Run test for each epsilon
for eps in epsilons:
    print("epsilon=",eps,':start test!')
    acc= test(model,device,test_loader,eps)
writer.close()





