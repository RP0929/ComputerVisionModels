

import os
#
#
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
print(torch.cuda.is_available())  # 查看cuda是否可用

print(torch.cuda.device_count())  # 返回GPU数目

print(torch.cuda.get_device_name(0))  # 返回GPU名称，设备索引默认从0开始

print(torch.cuda.current_device())  # 返回当前设备索引



# linux : nvidia-smi