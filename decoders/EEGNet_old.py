'''EEGNet prior to latency updates'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import CenterCrop
import pdb

name = 'EEGNet'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=0)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=(1,1), bias=bias)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ConstrainedDepthwiseConv2d(nn.Conv2d):
    def __init__(self, F1, D):
        super(ConstrainedDepthwiseConv2d, self).__init__(F1, F1*D, (64, 1), groups=F1)
        self._max_norm_val = 1
        self._eps = 0.01
    def forward(self, input):
        return F.conv2d(input, self._max_norm(self.weight), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

class ConstrainedDense(nn.Linear):
    def __init__(self, input_size=16, output_size=2):
        super(ConstrainedDense, self).__init__(input_size, output_size)
        self._max_norm_val = 0.25
        self._eps = 0.01
    def forward(self, input):
        return F.linear(input, self._max_norm(self.weight), self.bias)
    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

class EEGNet(nn.Module):

    def __init__(self, input_length=1, input_dim=63, output_dim=2, F1=8, D=2):
        super(EEGNet, self).__init__()

        F2 = F1 * D
        self.T = 1000  # timestamps in data
        self.C = 64    # electrodes in data

        self._downsample = nn.AvgPool2d((1,10)).to(device)  # downsamples from 1KHz to 100Hz

        # BLOCK 1 LAYERS
        self._input_crop = CenterCrop([self.C, 2000]).to(device)  # takes a 2 second window of data
        
        ##################################################
        # BLOCK 1
        ##################################################
        # 500ms temperal kernals
        self._conv1 = nn.Conv2d(1, F1, (1, 50), padding=0).to(device)   
        self._batchnorm1 = nn.BatchNorm2d(F1, False).to(device)

        ##################################################
        # BLOCK 2
        ##################################################
        self._depthwise = ConstrainedDepthwiseConv2d(F1, D).to(device)  
        self._batchnorm2 = nn.BatchNorm2d(F1*D).to(device)
        # reduce sampling rate to ~32Hz
        self._avg_pool = nn.AvgPool2d((1,3)).to(device) 
        self._dropout1 = nn.Dropout(p=0.5).to(device)

        # seperable convolution
        self._seperable = SeparableConv2d(F1*D, F2, (1,16)).to(device)  # ~500ms temporal windows
        self._batchnorm3 = nn.BatchNorm2d(F2).to(device)
        self._dropout2 = nn.Dropout(p=0.5).to(device)
        self._reduce_dimensions = nn.AvgPool2d((1,8)).to(device)
        self._dense = ConstrainedDense(input_size=12*F1, output_size=output_dim).to(device)

        # name of EEGNet model
        self._name = "EEGNet-" + str(F1) + "," + str(D)

    def get_name(self):
        return self._name

    def forward(self, x):
        # ensure input data is a tensor
        if isinstance(x, np.ndarray):  
            x = torch.FloatTensor(x)

        # input to model should be (batch_size, 1, electrodes, t_steps)
        X = self.reshape_input(x)

        # BLOCK 1
        #X = CenterCrop(X)
        X = self._downsample(X)
        X = self._conv1(X)
        X = self._batchnorm1(X)
        X = self._depthwise(X)
        X = self._batchnorm2(X)
        X = nn.ELU()(X)
        X = self._avg_pool(X)
        X = self._dropout1(X)

        # BLOCK 2
        #X = self._seperable(X)
        #X = self._batchnorm3(X)
        X = nn.ELU()(X)
        X = self._reduce_dimensions(X)
        X = self._dropout2(X)
        X = torch.flatten(X, start_dim=1)
        X = self._dense(X)
        X = nn.Softmax()(X)

        return X

    def reshape_input(self, x):
        '''
        x will be shape (1000, 64)
        Reshapes X to (1, 1, 64, 1000) which is batch_size, input_channels, electrodes, timesteps

        at inference the batch size will be 1
        the channel dimension is always 1 since EEG is interpreted as a greyscale image
        we only take 59 channels 
        we have a 64 chanel system
        and we have 1000 timesteps in a second
        '''
        if len(x.shape) == 2:   # x is (1000, 64)
            x = torch.transpose(x, 0, 1)#.to(device)   # x is now (64, 100)        # use for online
            #pdb.set_trace()
            #x = x.permute(0,3,1,2).to(device)              # use for offline
            x = self._input_crop(x)      # (59, 2000)
            x = torch.unsqueeze(torch.unsqueeze(x, 0), 0) # (1, 1, 59, 1000)   # use for online
        else:   # x is (batch_size, 1000, 64)
            x = torch.transpose(x, 1, 2)#.to(device)   # x is now (batch_size, 64, 100) 
            x = self._input_crop(x)      # (batch_size, 59, 1000)
            x = torch.unsqueeze(x, 1) # (batch_size, 1, 59, 1000)   # use for online
        return x
