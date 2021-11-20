# # # while True:
# # #     #其中evaluate_gradient是反向传播求得的梯度，loss_fun是前向传播的损失函数
# # #     #step_size是学习率
# # #     #the evaluate_gradient is the gradient which produced in backward prop
# # #     #the loss_fun means the loss function
# # #     #step_size is the learning rate
# # #     weights_grad = evaluate_gradient(loss_fun,data_weights)
# # #     weights += - step_size * weights_grad
# #
# #
# # #SGD
# # # while True:
# # #     dx = compute_gradient(x)
# # #     x -= learning_rate * dx
# #
# # #SGD+Momentum
# # # while True:
# # #     dx = compute_gradient(x)
# # #     vx = rho * vx + dx
# # #     x -= learning_rate * vx
# #
# # #Nesterov Momentum
# # # while True:
# # #      dx = compute_gradient(x)
# # #      old_v = v
# # #      v = roh*v -learning_rate * dx
# # #      x += -rho * old_v + (1+rho) * v
# #
# # #AdaGrad
# # import numpy as np
# #
# # # grad_squard = 0
# # # while True:
# # #     dx = compute_gradient(x)
# # #     grad_squared += dx * dx
# # #     # 惩罚项是越来越大的，则分母越来越大，更新的就会越来越少
# # #     x -= learning_rate * dx /(np.sqrt(grad_squared+1e-7))
# # #     # 1e-7是为了防止分母为0
# #
# # #RMSProp
# # grad_squared = 0
# # while True:
# #     dx = compute_gradient(x)
# #     grad_squared = decay_rate*grad_squared+(1-decay_rate)*dx*dx
# #     #decay_rate就是引入的衰减因子，反映保留多少我们之前的惩罚项
# #     #若decay_rate变为1，则更新项变成0
# #     #若decat_rate变为0，则不考虑之前的惩罚项
# #     x -= learning_rate * dx /(np.sqrt(grad_squared)+1e-7)
#
# # Adam(almost)
# import numpy as np
#
# # first_moment = 0
# # second_moment = 0
# # #上面的两个参数只有长时间训练才会真正发挥作用
# # while True:
# #     dx = compute_grandient(x)
# #     #第一动量
# #     first_moment = beta1*first_moment+(1-beta1)*dx
# #     #第二动量
# #     second_moment = beta2*second_moment+(1-beta2)*dx*dx
# #     x -= learning_rate * first_moment/(np.sqrt(second_moment)+1e-7)
#
# #Adam(full form)
# first_moment = 0
# second_moment = 0
# for i in range(1,num_iterations):
#     dx = compute_graient(x)
#     first_moment = beta1 * first_moment + (1 - beta1) * dx
#     second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
#     first_unbias = first_moment/(1-beta1**t)
#     second_unbias = second_moment/(1-beta2**t)
#     x -= learning_rate * first_unbias / (np.sqrt(second_unbias)+1e-7)