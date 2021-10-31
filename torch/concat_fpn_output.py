import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class concat_feature_maps(nn.Module):
    def __init__(self):
        super(concat_feature_maps, self).__init__()

    def forward(self, fpn_output):
        # Calculating median height to upsample or desample each fpn levels
        heights = []
        level_tensors = []
        for key, values in fpn_output.items():
            if key != 'pool':
                heights.append(values.shape[2])
                level_tensors.append(values)
        median_height = int(np.median(heights))

        # Upsample and Desampling tensors to median height and width
        for i in range(len(level_tensors)):
            level = level_tensors[i]
            # If level height is greater than median, then downsample with interpolate
            if level.shape[2] > median_height:
                level = F.interpolate(input=level, size=(median_height, median_height),mode='nearest')
            # If level height is less than median, then upsample
            else:
                level = F.interpolate(input=level, size=(median_height, median_height), mode='nearest')
            level_tensors[i] = level
        
        # Concating all levels with dimensions (batch_size, levels, C, H, W)
        concat_levels = torch.stack(level_tensors, dim=1)

        # Reshaping tensor from (batch_size, levels, C, H, W) to (batch_size, levels, HxW=S, C)
        concat_levels = concat_levels.flatten(start_dim=3).transpose(dim0=2, dim1=3)
        return concat_levels