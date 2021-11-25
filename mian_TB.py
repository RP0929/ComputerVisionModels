'''
Train CIFAR10 with PyTorch with TensorBoard.
'''
import os

from models.VGG.vgg import VGG16, VGG19

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# CUDA_VISIBLE_DEVICES=0
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
from models.VGG import *
from models.ResNet.resnet import *

from models import *
from TestFile.utils import progress_bar

writer = SummaryWriter()

#创建对象
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')  ###
parser.add_argument('--lr',default=0.1,type=float,help='learning rate')
parser.add_argument('--resume',default =False,action='store_true',help = 'resume from checkpoint',)
args = parser.parse_args()

device = 'cuda'
best_acc = 0
start_epoch = 0
total_train_step = 0
total_test_step = 0
#Data
print('==>Preparing CIFAR 10 data...')
transform_train = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',train=True,download=True,transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset,batch_size = 128,shuffle=True,num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data',train=False,download=True,transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset,batch_size = 128,shuffle = False,num_workers=2
)

classes = ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck')
#Module
print("==> Building models..")

net = resnet18()

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net) #!!!
    cudnn.benchmark = True


print(args.resume)
if args.resume==True:
    #Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'),'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_resnet18.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("checkpoint epoch:",start_epoch,"best_acc:",best_acc)

criterion = nn.CrossEntropyLoss()
#为了防止过拟合，在原本损失函数的基础上，加上L2正则化，而weight_decay就是这个正则化的lambda参数，一般设置为1e-8，所以调参的时候调整是否使用权重衰退即可。
#optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
#optimizer = optim.Adam(net.parameters(),args.lr,(0.9, 0.999),weight_decay=1e-4)
optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[135,185],0.1,-1)

#Training
def train(epoch):
    global total_train_step
    print('\nEpoch: %d'% epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        #TB
        if total_train_step%1000 == 0:
            writer.add_scalar("train_loss",loss.item(),total_train_step)
        total_train_step = total_train_step + 1

        train_loss += loss.item()
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx,len(trainloader),'Loss:%.3f | Acc:%.3f%%(%d/%d)'
                     %(train_loss/(batch_idx+1),100.*correct/total,correct,total))

def test(epoch):
    global best_acc,total_test_step
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx,(inputs,targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs,targets)

            test_loss += loss.item()

            _,predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,len(testloader),'Loss:%.3f | Acc:%.3f%% (%d/%d)'
                         %(test_loss/(batch_idx+1),100.*correct/total,correct,total))
    #Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar("test_acc", acc, total_test_step)
    total_test_step += 1
    if acc>best_acc:
        print("acc:",acc)
        print("Become Best!!! Saving Module now!")
        state = {
            'net':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,'./checkpoint/ckpt_resnet18.pth')
        best_acc = acc
    else:
        print("loss", test_loss)
        print("nowacc:",acc)
        print("Cheer up the bestacc is:",best_acc)

for epoch in range(start_epoch, start_epoch+240):
    train(epoch)
    test(epoch)
    scheduler.step()
writer.close()