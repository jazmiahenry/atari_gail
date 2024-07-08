"""Pytorch Implementation of Sharpened Cosine 2D."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SharpCosSim2D(nn.Module):
    """Model built based on sharpened cosine similarity logic.

    Args:
        kernel_size (int): Size of the convolutional kernel. Default is 1.
        units (int): Number of output units. Default is 32.
        input_shape (tuple): Shape of the input tensor. Default is (28, 28, 1).
    """
    
    def __init__(self, kernel_size=1, units=32, input_shape=(28, 28, 1)):
        super(SharpCosSim2D, self).__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        
        self.output_y = math.ceil(self.input_shape[0] / 1)
        self.output_x = math.ceil(self.input_shape[1] / 1)
        self.flat_size = self.output_x * self.output_y
        self.channels = self.input_shape[2]
        
        # Initialize weights and biases
        self.w = nn.Parameter(torch.randn(1, self.channels * self.kernel_size ** 2, self.units))
        nn.init.xavier_normal_(self.w)
        
        self.b = nn.Parameter(torch.zeros(self.units))
        self.p = nn.Parameter(torch.ones(self.units))
        self.q = nn.Parameter(torch.zeros(1))
    
    def l2_normal(self, x, axis=None, epsilon=1e-12):
        """Compute L2 normalization of a tensor.

        Args:
            x (torch.Tensor): Input tensor.
            axis (int or tuple of int, optional): Axis or axes along which to compute the normalization.
            epsilon (float, optional): Small value to avoid division by zero. Default is 1e-12.

        Returns:
            torch.Tensor: L2 normalized tensor.
        """
        square_sum = torch.sum(x ** 2, dim=axis, keepdim=True)
        x_inv_norm = torch.sqrt(torch.clamp(square_sum, min=epsilon))
        return x_inv_norm
    
    def forward(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying sharpened cosine similarity.
        """
        # Reshape the input tensor
        x = inputs.view(-1, self.flat_size, self.channels * self.kernel_size ** 2)
        
        # Normalize the input and weight tensors
        x_norm = self.l2_normal(x, axis=2)
        w_norm = self.l2_normal(self.w, axis=1)
        
        # Compute cosine similarity
        x = torch.matmul(x / x_norm, self.w / w_norm)
        
        # Apply sharpening function
        sign = torch.sign(x)
        x = torch.abs(x) + 1e-12
        x = torch.pow(x + self.b ** 2, torch.sigmoid(self.p) * F.softplus(self.p))
        x = sign * x
        
        # Reshape the output tensor to match the expected output shape
        x = x.view(-1, self.output_y, self.output_x, self.units)
        return x
