#VGG16:2->2->3->3->3
#VGG19:2->2->4->4->4
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,layer_num,in_planes,planes):
        super(BasicBlock, self).__init__()
        self.block = self.make_layers(layer_num,in_planes,planes)

    def make_layers(self,layer_num,in_planes,planes):
        layers = []
        for i in range(layer_num):
            if i == 0:
                layers.append(nn.Conv2d(in_planes,planes,kernel_size=3,stride=1,padding=1))
                layers.append(nn.BatchNorm2d(planes)),
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(planes)),
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    def forward(self,x):
        out = self.block(x)
        return out

class VGG(nn.Module):
    def __init__(self,num_blocks,num_class=10):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.block1 = self.make_layers(num_blocks[0],64,64)
        self.block2 = self.make_layers(num_blocks[1],64,128)
        self.block3 = self.make_layers(num_blocks[2],128,256)
        self.block4 = self.make_layers(num_blocks[3],256,512)
        self.block5 = self.make_layers(num_blocks[4],512,512)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(512,10)
        #self.linear2 = nn.Linear(256,num_class)
    def make_layers(self,layer_nums,in_planes,planes):
        return BasicBlock(layer_nums,in_planes,planes)
    def forward(self,x):
        out = self.conv1(x)
        #print("conv1:", out.shape)
        out = self.block1(out)
        #print("b1:",out.shape)
        out = self.maxpool(out)
        #print('mp1:',out.shape)
        out = self.block2(out)
        #print("b2:", out.shape)
        out = self.maxpool(out)
        #print('mp2:', out.shape)
        out = self.block3(out)
        #print("b3:", out.shape)
        out = self.maxpool(out)
        #print('mp3:', out.shape)
        out = self.block4(out)
        #print("b4:", out.shape)
        out = self.maxpool(out)
        #print("mp4:", out.shape)
        out = self.block5(out)
        #print("b5:", out.shape)
        out = self.maxpool(out)
        #print("mp5:", out.shape)
        out = out.view(out.size(0),-1)
        #print('1',out.size())
        #print("flatten:",out.shape)
        out = self.linear1(out)
        # out = self.linear2(out)
        return out

def test():
    x = torch.randn(64,3,32,32)
    net = VGG16()
    out = net(x)
    print(out.shape)

def VGG16():
    num_blocks = [2, 2, 3, 3, 3]
    net = VGG(num_blocks, 10)
    return net
def VGG19():
    num_blocks = [2, 2, 4, 4, 4]
    net = VGG(num_blocks, 10)
    return net
#test()