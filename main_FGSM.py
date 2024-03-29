import os

import torchvision.datasets
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
writer = SummaryWriter('./adv/adversarial_samples_FGSM_BatchSize/')
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

#FGSM attack code
def fgsm_attack(image,epsilon,data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    if epsilon !=0 :
        perturbed_image = torch.clamp(perturbed_image,0,1)
    return perturbed_image

def test(model,device,test_loader,epsilon):
    # Accuracy counter
    correct = 0
    total = 0
    global epoch,image_num
    image_step = 0
    epoch += 1
    #Loop over all examples in test set
    for batch_idx,(data,target) in enumerate(test_loader):
        # Send the data and label to the device
        data,target = data.to(device),target.to(device)
        # Set requires_grad attribute of tensor. This is important for Attack
        data.requires_grad = True
        # Forward pass the data through the model

        total += target.size(0)

        outputs = model(data)
        init_pred = outputs.max(1,keepdim=True)[1]
        loss = F.nll_loss(outputs, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()
        #print("init_pred size",init_pred.squeeze(1).size(),init_pred.squeeze(1))
        #print("target size",target.size(),target)
        #print("total size",torch.eq(init_pred.squeeze(1),target).size(),torch.eq(init_pred.squeeze(1),target))

        #print(batch_idx,":",torch.eq(init_pred.squeeze(1),target).sum().item())
        #if the initial prediction is wrong,dont bother attacking,just move on
        if torch.eq(init_pred,target).sum().item()==0 :
            continue

        #Caculate datagrad
        data_grad = data.grad.data

        #Call FGSM Attack
        perturbed_data = fgsm_attack(data,epsilon,data_grad)
        if batch_idx==9:
            print("adding now!")
            writer.add_images("Origanel",data,epoch)
            writer.add_images("Adv Image",perturbed_data,epoch)


        outputs = model(perturbed_data)
        # Calculate the loss

        # Check for success
        final_pred = outputs.max(1, keepdim=True)[1]
        #if final_pred.item() == target.item():
            #correct += 1
            #Special case for saving 0 epsilon examples


        correct += torch.eq(final_pred.squeeze(1),target).sum().item()

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





