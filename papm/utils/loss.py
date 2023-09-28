import torch


class Loss_fun():
    def __init__(self):
        pass

    def relative_loss(self, output, target):
        error = output - target
        norm_error_channel = torch.norm(error, dim=[3, 4]) / (torch.norm(target, dim=[3, 4]) + 1e-8)
        norm_error_time = norm_error_channel.mean(dim=2)
        acc = norm_error_time.mean(dim=[0, 1])
        return acc

    # return array, length=timesteps
    def point_relative_loss(self, output, target):    
       # 计算误差
        error = output - target
        norm_error_channel = torch.norm(error, dim=[3, 4]) / (torch.norm(target, dim=[3, 4]) + 1e-8)
        norm_error_time = norm_error_channel.mean(dim=2)
        acc = norm_error_time.mean(dim=0)
        return acc
