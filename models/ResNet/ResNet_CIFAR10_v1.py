#要想指定cuda需要将2,3行代码放在import torch之前
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# 加载CIFAR10数据集
batchsz = 10
cifar_train = datasets.CIFAR10("./data/",True,transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
]),download=True)
cifar_train = DataLoader(cifar_train,batch_size=batchsz,shuffle=True)
cifar_test = datasets.CIFAR10("./data/", True, transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
]), download=True)
cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
X,label = iter(cifar_train).next()
print('x:',X.shape,'label:',label.shape)

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        #上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
        self.relu = nn.ReLU(inplace=True)

    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


#ResNet模型 b1
b1 = nn.Sequential(
    nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),#说明输入图像的通道数为3，第一输出层通道数为64，卷积核边长为7，padding=3，stride=2 ，
    nn.BatchNorm2d(64),#在使用pytorch的 nn.BatchNorm2d() 层的时候，经常地使用方式为在参数里面只加上待处理的数据的通道数（特征数量）
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)#为什么？
) #论文中34层带残差的网络第1部分

#b2、b3、b4、b5都具有相同的残差块结构
def resnet_block(input_channels,num_channels,num_resduals,first_block=False):#针对于第一个卷积层，需要用1×1卷积进行升维处理，numresidual从0开始所以实际的卷积层是指定值+1
    blk = []
    for i in range(num_resduals):
        if i == 0 and not first_block:
            #为什么要在这一步进行变化？
            blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides = 2))#/2表示下采样（即步长为2的卷积），要注意第一个卷积层不仅通道变化，图像的长宽也发生了变化
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))#resnet_block返回的是一个残差块列表，*就是把列表展开
b3 = nn.Sequential(*resnet_block(64,128,2))#stride = 2 那么图像会变小
b4 = nn.Sequential(*resnet_block(128,256,2))
b5 = nn.Sequential(*resnet_block(256,512,2))
device = torch.device('cuda')
net = nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((5,5)),nn.Flatten(),nn.Linear(12800,10)).to(device)#nn.Flatten是拉平 #为什么一加入todevice就错了

#nn.AdaptiveAvgPool2d???
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(),lr=1e-3)
for epoch in range(10):
    for batchidx,(x,label) in enumerate(cifar_train):
        x , label = x.to(device),label.to(device)
        y_pred = net(x)
        loss = criteon(y_pred,label)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch,loss.item())
    net.eval()
    with torch.no_grad():
        #测试集测试
        total_correct = 0
        total_num = 0
        for batchidx,(x,label) in enumerate(cifar_test):
            x, label = x.to(device), label.to(device)
            y_pred = net(x)
            correct = torch.eq(y_pred,label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
        acc = total_correct / total_num
        print(epoch,"test acc:",acc)
#经过b2层可知图像大小变为 1*64*56*56         卷积层：（224-7+2）/ 2 + 1 =  111 （111-3+2）/2+1 = 56，通道数为64
#经过b3层可知图像大小变为 1*128*28*28        卷积层：（56-3+2）/2 + 1 = 27+1 = 28 （28-3+2）/1+1 = 28，通道数为128
#经过b4层可知图像大小变为 1*256*14*14        卷积层；（28-3+2）/2 + 1 = 14 (14 - 3 + 2) / 1 + 1 = 14,通道数位256
#经过b6层可知图像大小变为 1*512*7*7          卷积层；（14-3+2）/2 + 1 = 7 (7 - 3 + 2) / 1 + 1 = 7,通道数位512
#经过平均池化层：nn.AdaptiveAvgPool2d((1,1))，其参数表示输出图像的大小，即变为了1*512*1*1的情况
#nn.Flatten()全部展开
#nn.Linear(512,10)经过线性层






