import torch


class Loss_fun():
    def __init__(self):
        pass

    def channel_relative_error(output, target, eps=1e-8):
        """
        Compute the relative error between output and target at the channel level.
    
        Parameters:
        - output: Predicted tensor.
        - target: Ground truth tensor.
        - eps: A small number to prevent division by zero.
    
        Returns:
        - Relative error at each time step after averaging over channels.
        """
        error = output - target
        norm_error_channel = torch.norm(error, dim=[3, 4]) / (torch.norm(target, dim=[3, 4]) + eps)
        norm_error_time = norm_error_channel.mean(dim=2)
        return norm_error_time

    def relative_loss(output, target, alpha):
        """
        Compute the weighted loss using relative error at the channel level.
    
        Parameters:
        - output: Predicted tensor.
        - target: Ground truth tensor.
        - alpha: Weighting factor.
    
        Returns:
        - Weighted loss value.
        """
        # Ensure that the input shapes are correct.
        assert output.shape == target.shape, "Output and Target shapes do not match!"
        
        # Calculate the relative error for each time step.
        basic_loss = channel_relative_error(output, target)
        
        # Calculate weights w_i and detach to ensure no gradient information is carried.
        cum_loss = torch.cumsum(basic_loss, dim=1)
        w = torch.exp(-alpha * torch.cat([torch.zeros(basic_loss.size(0), 1).to(output.device), cum_loss[:, :-1]], dim=1))
        w = w.detach()
        
        # Compute the final weighted loss using the weights.
        weighted_loss = w * basic_loss
        final_loss = weighted_loss.mean()
    
        return final_loss

    # return array, length=timesteps
    def point_relative_loss(self, output, target):
        # error
        error = output - target
        norm_error_channel = torch.norm(error, dim=[3, 4]) / (torch.norm(target, dim=[3, 4]) + 1e-8)
        norm_error_time = norm_error_channel.mean(dim=2)
        acc = norm_error_time.mean(dim=0)
        return acc
