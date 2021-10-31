import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class Scale_Aware_Layer(nn.Module):
    # Constructor
    def __init__(self, s_size):
        super(Scale_Aware_Layer, self).__init__()

        # Average Pooling
        self.avg_layer = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        #1x1 Conv layer
        self.conv = nn.Conv2d(in_channels=s_size, out_channels=1, kernel_size=1)

        # Hard Sigmoid
        self.hard_sigmoid = nn.Hardsigmoid()

        # ReLU function
        self.relu = nn.ReLU()

    def forward(self, F):

        # Transposing input from (batch_size, L, S, C) to (batch_size, S, L, C) so we can use convolutional layer over the level dimension L
        x = F.transpose(dim0=2, dim1=1)
        
        # Passing tensor through avg pool layer
        x = self.avg_layer(x)

        # Passing tensor through Conv layer
        x = self.conv(x)

        # Reshaping Tensor from (batch_size, 1, L, C) to (batch_size, L, 1, C) to then be multiplied to F
        x = x.transpose(dim0=1, dim1=2)

        # Passing conv output to relu
        x = self.relu(x)

        # Passing tensor to hard sigmoid function
        pi_L = self.hard_sigmoid(x)

        # pi_L: (batch_size, L, 1, C)
        # F: (batch_size, L, S, C)
        return pi_L * F

class Spatial_Aware_Layer(nn.Module):
    # Constructor
    def __init__(self, L_size, kernel_height=3, kernel_width=3, padding=1, stride=1, dilation=1, groups=1):
        super(Spatial_Aware_Layer, self).__init__()

        self.in_channels = L_size
        self.out_channels = L_size

        self.kernel_size = (kernel_height, kernel_width)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.K = kernel_height * kernel_width
        self.groups = groups

        # 3x3 Convolution with 3K out_channel output as described in Deform Conv2 paper
        self.offset_and_mask_conv = nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=3*self.K, #3K depth
                                              kernel_size=self.kernel_size,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=dilation)
        
        self.deform_conv = DeformConv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=self.groups)
    def forward(self, F):
        # Generating offesets and masks (or modulators) for convolution operation
        offsets_and_masks = self.offset_and_mask_conv(F)

        # Separating offsets and masks as described in Deform Conv v2 paper
        offset = offsets_and_masks[:, :2*self.K, :, :] # First 2K channels 
        mask = torch.sigmoid(offsets_and_masks[:, 2*self.K:, : , :]) # Last 1K channels and passing it through sigmoid

        # Passing offsets, masks, and F into deform conv layer
        spacial_output = self.deform_conv(F, offset, mask)
        return spacial_output

# DyReLUA technique from Dynamic ReLU paper
class DyReLUA(nn.Module):
    def __init__(self, channels, reduction=8, k=2, lambdas=None, init_values=None):
        super(DyReLUA, self).__init__()

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels//reduction, 2*k)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Defining lambdas in form of [La1, La2, Lb1, Lb2]
        if lambdas is not None:
            self.lambdas = lambdas
        else:
            # Default lambdas from DyReLU paper
            self.lambdas = torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float)

        # Defining Initializing values in form of [alpha1, alpha2, Beta1, Beta2]
        if lambdas is not None:
            self.init_values = init_values
        else:
            # Default initializing values of DyReLU paper
            self.init_values = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float)

    def forward(self, F_tensor):

        # Global Averaging F
        kernel_size = F_tensor.shape[2:] # Getting HxW of F
        gap_output = F.avg_pool2d(F_tensor, kernel_size)

        # Flattening gap_output from (batch_size, C, 1, 1) to (batch_size, C)
        gap_output = gap_output.flatten(start_dim=1)

        # Passing Global Average output through Fully-Connected Layers
        x = self.relu(self.fc1(gap_output))
        x = self.fc2(x)
        
        # Normalization between (-1, 1)
        residuals = 2 * self.sigmoid(x) - 1

        # Getting values of theta, and separating alphas and betas
        theta = self.init_values + self.lambdas * residuals # Contains[alpha1(x), alpha2(x), Beta1(x), Beta2(x)]
        alphas = theta[0, :2]
        betas = theta[0, 2:]

        # Performing maximum on both piecewise functions
        output = torch.maximum((alphas[0] * F_tensor + betas[0]), (alphas[1] * F_tensor + betas[1]))

        return output

class Task_Aware_Layer(nn.Module):
    # Defining constructor
    def __init__(self, num_channels):
        super(Task_Aware_Layer, self).__init__()

        # DyReLUA relu
        self.dynamic_relu = DyReLUA(num_channels)
    
    def forward(self, F_tensor):
        # Permutating F from (batch_size, L, S, C) to (batch_size, C, L, S) so we can reduce the dimensions over LxS
        F_tensor = F_tensor.permute(0, 3, 1, 2)

        output = self.dynamic_relu(F_tensor)

        # Reversing the permutation
        output = output.permute(0, 2, 3, 1)

        return output