import os

import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from models.ResNet.resnet import *
## cudnn???
import torch.backends.cudnn as cudnn

device = 'cuda'
epsilons = [0,.0025,.005,.015,.025,.05,.075,.1,.15,.2,.25,.3,.35,.4,.5]
pretranined_model = './checkpoint/ckpt_resnet18.pth'
use_cuda = True
checkpoint = torch.load(pretranined_model)
writer = SummaryWriter('./adv/adversarial_samples_FGSM_SM_NoNorm/')
print('==>Preparing CIFAR 10 data...')
#transform

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# CIFAR10 dataset and dataloader declaration
test_set = torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False,num_workers=8)
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
    # if epsilon !=0 :
    #     perturbed_image = torch.clamp(perturbed_image,-1,1)
    return perturbed_image

def test(model,device,test_loader,epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    global epoch,image_num
    image_step = 0
    epoch += 1
    #Loop over all examples in test set
    for data,target in test_loader:
        # Send the data and label to the device
        data,target = data.to(device),target.to(device)
        # Set requires_grad attribute of tensor. This is important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        outputs = model(data)
        init_pred = outputs.max(1,keepdim=True)[1]

        #if the initial prediction is wrong,dont bother attacking,just move on
        if init_pred.item() != target.item():
            continue

        #Calculate the loss
        loss = F.nll_loss(outputs,target)

        #Zero all existing gradients
        model.zero_grad()

        #Calculate gradients of model in backward pass
        loss.backward()

        #Caculate datagrad
        data_grad = data.grad.data

        #Call FGSM Attack
        perturbed_data = fgsm_attack(data,epsilon,data_grad)
        image_step += 1
        if image_step  == 800:
            print("add epoch",epoch,'No. pic:',image_step)
            image_num += 1
            #print("add original Img")
            writer.add_images("Original",data,image_num)
            #print("add Img")
            writer.add_images("Adv Image", perturbed_data,image_num)

        #Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1,keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            #Special case for saving 0 epsilon examples
            if epsilon == 0 :
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(),final_pred.item(),adv_ex))
    #Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    writer.add_scalar("Attack Acc:",final_acc,epoch)
    print("Epsilon: {}\tTest Accuracy = {}/{}={}".format(epsilon,correct,len(test_loader),final_acc))
    return final_acc,adv_examples

#Run Attack
accuracies = []
examples = []

#Run test for each epsilon
for eps in epsilons:
    print("epsilon=",eps,':start test!')
    acc,ex = test(model,device,test_loader,eps)
    accuracies.append(acc)
    examples.append(ex)
writer.close()





