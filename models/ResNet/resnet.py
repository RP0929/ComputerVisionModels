#import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pdb
# pdb.set_trace()
#define BasicBlock class(2func:init forward)

class BasicBlock(nn.Module):
    #the first param : whether explansion channels
    #in_planes/planes/kernel_size/padding/stride
    expansion = 1

    def __init__(self,in_planes,planes,kernel_size=3,padding=1,stride=1):
        #must have super!!!
        super(BasicBlock, self).__init__()
        #the position of params is inplanes,planes,kernel_size,padding,stride,bias
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size,padding=padding,stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size,padding=padding,stride=1) #after key argument we cannot use param without clear val.
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes*self.expansion:
            # in this case we have to use 1*1 conv to make same wideth feature_map or same channel feature_map
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self,x): #x is input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




#test()
#define Bottleneck class(2func:init forward) after!
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_planes,planes,stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,stride=stride,padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        #shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or planes!=in_planes*self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,padding=0,stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        return F.relu(out)
#define ComputerVisionModels class(3 func:init _make_layer forward)
class ResNet(nn.Module):

 #__init__ need the param to let user choose result num_class
    def __init__(self,block,num_blocks,num_class=10):
        super(ResNet, self).__init__()
        # block means type of block
        # num_blocks means how many layers in each block
        # num_class is the output results.It's 10 on CIFAR10
        self.inplanes = 64
        self.layer1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpooling = nn.MaxPool2d(3,2,padding=1)
        #max pooling layer don't need write in init because it's not the params layer
        self.layer2 = self._make_layer(block,64,num_blocks[0],stride=1)#before conv we use stride=2 maxpooling
        self.layer3 = self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer4 = self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer5 = self._make_layer(block,512,num_blocks[3],stride=2)#I think may be 1 or 2 all can but false
        #Avg pooling layer don't need write in init because it's not the params layer
        self.linear = nn.Linear(512*block.expansion,num_class)
    pass
 #_make_layer:block,planes,num_blocks,stride
    def _make_layer(self,block,planes,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layer = []
        for i in range(num_blocks):#range donot include the num_blocks
            layer.append(block(self.inplanes,planes,stride = strides[i]))
            #print("before--->>inplanes:", self.inplanes, "planes:", planes)
            self.inplanes = block.expansion * planes
            #print("after-->>inplanes:", self.inplanes, "planes:", planes)
        return nn.Sequential(*layer)
 # init param:block,num_blocks,num_classes
    def forward(self,x):
        out = F.relu(self.bn1(self.layer1(x)))
        out = self.maxpooling(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out
 #forward

def resnet18():
    net = ResNet(BasicBlock,[2,2,2,2])
    return net
def testBasicResNet():
    x = torch.randn(1, 3, 224, 224)
    net = resnet18()
    out = net(x)
    return out

#testBasicResNet()

def resnet50():
    net = ResNet(Bottleneck,[3,4,6,3])
    return net

def testBottleneck():
    x = torch.randn(1,3,112,112)
    net = resnet50()
    out = net(x)
    return out

net = testBottleneck()
print(net.size())