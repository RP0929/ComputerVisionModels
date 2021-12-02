import torch
import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import *
import numpy as np

class PGD(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.device = torch.device("cuda:0,1")
    def generate(self,x,**params):
        self.parse_params(**params)
        labels = self.y
        adv_x = self.attack(x,labels)
        return adv_x
    def parse_params(self,eps=0.3,iter_eps=0.01,nb_iter=40,clip_min=0.0,clip_max=1.0,
                     C=0.0,y=None,ord=np.inf,rand_init=True,flag_target=False):
        self.eps = eps
        self.iter_eps = iter_eps
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y = y
        self.ord = ord
        self.rand_init = rand_init
        self.model.to(self.device)
        self.flag_target = flag_target
        self.C = C

    def single_step_attack(self,x,pertubation,labels):
        adv_x = x+pertubation
        # get the gradient of x
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True
        loss_func = nn.CrossEntropyLoss()
        preds = self.model(adv_x)
        if self.flag_target:
            loss = - loss_func(preds,labels)
        else:
            loss = loss_func(preds,labels)
        self.model.zero.grad()
        loss.backward()
        grad = adv_x.grad.data
        #get the pertubation of an iter_eps
        pertubation = self.iter_eps*np.sign(grad)
        adv_x = adv_x.cpu().detach().numpy()+pertubation.cpu().numpy()
        x = x.cpu().detach().numpy()
        pertubation = np.clip(adv_x,self.clip_min,self.clip_max)-x
        pertubation = np.clip.c