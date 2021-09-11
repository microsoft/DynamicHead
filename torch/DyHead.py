import torch.nn as nn
from attention_layers import Scale_Aware_Layer, Spatial_Aware_Layer, Task_Aware_Layer
from collections import OrderedDict

class DyHead_Block(nn.Module):
    def __init__(self, L, S, C):
        super(DyHead_Block, self).__init__()
        # Saving all dimension sizes of F
        self.L_size = L
        self.S_size = S
        self.C_size = C

        # Inititalizing all attention layers
        self.scale_attention = Scale_Aware_Layer(s_size=self.S_size)
        self.spatial_attention = Spatial_Aware_Layer(L_size=self.L_size)
        self.task_attention = Task_Aware_Layer(num_channels=self.C_size)

    def forward(self, F_tensor):
        scale_output = self.scale_attention(F_tensor)
        spacial_output = self.spatial_attention(scale_output)
        task_output = self.task_attention(spacial_output)

        return task_output

def DyHead(num_blocks, L, S, C):
    blocks = [('Block_{}'.format(i+1),DyHead_Block(L, S, C)) for i in range(num_blocks)]

    return nn.Sequential(OrderedDict(blocks))
