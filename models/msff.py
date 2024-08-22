import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .coordatt import CoordAtt

#  an intuitive idea is to fully fuse the
# visual information and semantic information in the concatenated information  with the help of the channel attention
# mechanism and multi-scale feature fusion strategy
class MSFFBlock(nn.Module):
    def __init__(self, in_channel):
        super(MSFFBlock, self).__init__()
        # d by a 3×3 convolutional layer that maintains the number of channels
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        # attetnion
        self.attn = CoordAtt(in_channel, in_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x_conv = self.conv1(x)
        # these are the 2 parallel branchses in the figure
        x_att = self.attn(x)
        # you merge the eatention
        x = x_conv * x_att
        # two times attention exactly as they menitond

        # this is the attneiton that is mentioned to allgin the different multi scale feature
        # at different levels
        # you need to make sure you have aligned number of channels in here
        x = self.conv2(x)
        return x


class MSFF(nn.Module):
    def __init__(self):
        super(MSFF, self).__init__()
        # Layer 1 ---> torch.Size([8, 64, 64, 64]) FROZEN LAYER
        # Layer 2 ---> torch.Size([8, 128, 32, 32]) FROZEN LAYER
        # Layer 3 ---> torch.Size([8, 256, 16, 16]) FROZEN LAYER
        self.blk1 = MSFFBlock(128)
        self.blk2 = MSFFBlock(256)
        self.blk3 = MSFFBlock(512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #  by a 3×3 convolutional layer that maintains the number of channels.
        self.upconv32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        )
        self.upconv21 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, features):
        # features = [level1, level2, level3]
        f1, f2, f3 = features 
        
        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        f2_f = f2_k + self.upconv32(f3_k)
        f1_f = f1_k + self.upconv21(f2_f)

        # spatial attention
        
        # mask
        '''
        . f3[:, 256:, ...]:
f3: This is likely a 4D tensor (common in image processing tasks, especially with batches of images).
:: This selects all elements along the first axis (often the batch dimension).
256:: This selects elements from index 256 to the end along the second axis (often the height or width of an image).
...: This is a shorthand that means "all remaining axes". It allows you to easily select slices without specifying every dimension explicitly.
Together, f3[:, 256:, ...] selects a subset of the tensor f3, specifically from the 256th index onward in the second dimension and all elements in the other dimensions.

2. .mean(dim=1, keepdim=True):
.mean(): This function calculates the mean (average) across a specified dimension of the tensor.
dim=1: This specifies the dimension along which to calculate the mean. Since PyTorch uses zero-based indexing, dim=1 refers to the second axis of the tensor (often corresponding to the height of an image, in typical image processing tensors).
keepdim=True: This ensures that the reduced dimension (dimension 1) is kept in the result as a dimension with size 1, instead of being removed. This helps to maintain the original tensor's shape with just that specific dimension reduced to size 1.
        '''
        m3 = f3[:,256:,...].mean(dim=1, keepdim=True)
        m2 = f2[:,128:,...].mean(dim=1, keepdim=True) * self.upsample(m3)
        m1 = f1[:,64:,...].mean(dim=1, keepdim=True) * self.upsample(m2)
        
        f1_out = f1_f * m1
        f2_out = f2_f * m2
        f3_out = f3_k * m3
        
        return [f1_out, f2_out, f3_out]