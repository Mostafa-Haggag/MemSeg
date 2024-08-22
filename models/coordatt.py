# https://github.com/houqb/CoordAttention/blob/main/coordatt.py

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    '''
    hard sigmoid
    '''
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        #  which is a modified ReLU activation that clips the input to the range [0,6]
        #  if set to True, modifies the input directly without allocating additional memory.
    def forward(self, x):
        # shift the input by 3 unites
        # apply th relu 6 so that it is in range of 0 to 6
        # divide by 6 to clip between 0 to 1
        # This effectively implements a hard approximation of the sigmoid function,
        # which is faster and simpler to compute.
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    '''
     This class defines the hard swish (h-swish) activation function.
     # This creates a smooth, non-linear activation that is similar to the Swish function,
      but with a piecewise linear approximation, making it more computationally efficient.
    '''
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        # It initializes an instance of the previously defined h_sigmoid class.
        self.sigmoid = h_sigmoid(inplace=inplace)
        # the orginal input is multpedi by the value output from hard sard sigmoid to get a resulting
        # in an output range of [0,x] when x>0 and [-x,0] when x < 0
    def forward(self, x):
        # input passed to hard sigmoid to get output of 0 to 1
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    '''
    # to simply understand what is happening
    you need to check this https://blog.paperspace.com/coordinate-attention/
    **
    The code defines a custom PyTorch module called CoordAtt, which stands for Coordinate Attention.
    This module is a form of attention mechanism that enhances the network's ability to capture spatial
    information by emphasizing the importance of different spatial dimensions (height and width) separately.
    ** The CoordAtt module is designed to improve the representation of spatial features in convolutional
    neural networks (CNNs). Here's a breakdown of the ideas and components used in this module:
    ##### Spatial Attention Mechanism
        The primary idea behind the CoordAtt module is to focus on the spatial dimensions (height and width) separately,
        allowing the network to better capture spatial dependencies in the feature maps.
        This is particularly useful in tasks where the spatial structure of the input data is crucial,
        such as object detection and segmentation.
    # Sumary
    CoordAtt is a coordinate attention mechanism that separately models attention for height and width,
    allowing the network to better capture spatial dependencies in the feature maps.
    The adaptive pooling and separate convolutions for height and width help
    the network emphasize important spatial features in both dimensions,
    leading to improved performance on tasks where spatial information is critical.
    By using a lightweight architecture
     (e.g., reduction in dimensionality and efficient operations like convolution and pooling),
      CoordAtt can be integrated into existing CNN architectures with minimal computational overhead.
    '''

    def __init__(self, inp, oup, reduction=32):
        # inp and oup are the same value
        # 128 for first layer [8, 64, 64, 64]
        # 256 for second layer [8, 128, 32, 32]
        # 512 for third layer [8, 256, 16, 16]
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        #FYI for adaptive average pooling
        # AdaptiveAvgPool2d is a layer in PyTorch that applies 2D adaptive average pooling to input tensors.
        # This operation reduces the spatial dimensions (height and width) of the input tensor to a specified
        # target size by computing the average value over the input regions.
        #  AdaptiveAvgPool2d directly specifies the output size. The layer automatically determines the size of
        #  the pooling regions (kernels) to ensure that the input tensor is reduced to the desired output size.
        # FYI    # A tuple (target_height, target_width), which dictates the exact dimensions of the output.
        # FYI    # An integer, target_size, which applies the same target size to both dimensions
        # FYI    # (i.e., target_height = target_width = target_size).
        # FYI    # If None is specified for one of the dimensions, it will keep that dimension unchanged.
        # : Pools the input along the height dimension, resulting in a feature map of size (n,c,h,1).
        # what are the advantage in here?
        # The key advantage of this adaptive pooling is that it allows you to reduce an input tensor to a fixed size
        # regardless of its original size, which is useful when you want to feed data of varying sizes into layers
        # that require a fixed-size input.
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # : Pools the input along the width dimension, resulting in a feature map of size (n,c,1,w).
        # These pooled features retain global information about the height and width dimensions separately.
        mip = max(8, inp // reduction)
        # you choose between 8 and
        # For layer 1 64//32 = 2
        # For layer 1 128//32 = 4
        # For layer 1 256//32 = 8
        # you set mip to 8 for all of them
        #conv with kernel 1 is used to reduce the dimension fo channels
        # you will go down from previous layout to 8
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        # you are going up in the number of features maps from 8 to rginal size
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x) # the shape in here is (n,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # the output shape in here is (n,c,1,w)
        # we do the permute operation to switch to  (n,c,w,1)

        y = torch.cat([x_h, x_w], dim=2)
        # After pooling, the feature maps are concatenated along the spatial dimension (height and width),
        # creating a combined feature map.
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2) # you are splitting after concating them
        x_w = x_w.permute(0, 1, 3, 2) # returning everything back to (n,c,1,w)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        # you return to orginal number of layout then you appply sigmoid
        # these are the activations for the height
        # these are the attention activatiosn for the height and for the width
        out = identity * a_w * a_h

        return out