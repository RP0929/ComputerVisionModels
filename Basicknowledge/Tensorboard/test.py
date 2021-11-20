from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# writer.add_image()
# global_step对应的是x轴，scalar_value对应的是y轴
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()