'''Train CIFAR10 with PyTorch.'''
import os

from models.VGG.vgg import VGG16, VGG19

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# CUDA_VISIBLE_DEVICES=0
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
from models.VGG import *
from models.ResNet.resnet import *

from models import *
from TestFile.utils import progress_bar

#创建对象
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')  ###
#可以通过add_argument来进行参数的设置
#nargs表示使用参数的取值，+表示至少有一个参数，？就是0个或1个，*表示0个或所有
#type表示传入参数类型，有str,float,int类型
#default表示当你没有使用这个值的时候就使用默认值
#help是用来完成帮助信息
#action有store_true->false store_false->true action  原因表示活动，只有在具有触发动作时才显示作用，所以 store_xxx 后面的 xxx
#（true/false）表示的是触发后的参数值；default 表示默认参数值，可对应不触发 action 时的参数值，所以通常来讲 default=False 和
#action='store_true' 会成对出现，default=True 和 action='store_false' 会成对出现 ，最终实现既有参数默认功能，又有参数触发切换功能。
parser.add_argument('--lr',default=0.1,type=float,help='learning rate')
parser.add_argument('--resume',default =False,action='store_true',help = 'resume from checkpoint')
args = parser.parse_args()
#after this step:we can use args.lr args.resume

device = 'cuda'
best_acc = 0
start_epoch = 0

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
optimizer = optim.SGD(net.parameters(),lr=args.lr,momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)

#Training
def train(epoch):
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

        train_loss += loss.item()
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx,len(trainloader),'Loss:%.3f | Acc:%.3f%%(%d/%d)'
                     %(train_loss/(batch_idx+1),100.*correct/total,correct,total))

def test(epoch):
    global best_acc
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
    if acc>best_acc:
        print("test loss:",test_loss)
        print("acc:",acc)
        print("Become Best!!! Saving Module now!")
        state = {
            'net':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,'./checkpoint/ckpt_resnet18_avgk_2.pth')
        best_acc = acc
    else:
        print("loss", test_loss)
        print("nowacc:",acc)
        print("Cheer up the bestacc is:",best_acc)

for epoch in range(start_epoch, start_epoch+200):
    train(epoch+1)
    test(epoch)
    scheduler.step()