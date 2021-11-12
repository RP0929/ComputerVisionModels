import torch
from torch import nn

inputs = torch.tensor([1,2,3],dtype=torch.float32)
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss = nn.L1Loss()
result = loss(inputs,targets)
print(result)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs,targets)
print(result_mse)

