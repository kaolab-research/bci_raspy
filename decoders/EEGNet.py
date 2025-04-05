# Linear network for inputs of multiple time samples.
# Expects input of shape (batch_size, input_length, input_dim) or (input_length, input_dim) or (input_dim,) only if input_length = 1
# Output has shape (batch_size, output_dim). batch_size = 1 if not given.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms.functional import crop
import pdb
from dataclasses import dataclass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

name = 'EEGNet'

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

@dataclass
class EEGNetOutput:
    '''class to handle EEGNet outputs'''
    probs: torch.Tensor # softmax output
    logits: torch.Tensor # logits (pre-softmax)
    hidden_state: torch.Tensor # hidden_state (pre-dense)
    pass

class EEGNet(nn.Module):
    '''a comment'''
    def __init__(self, output_dim=2, F1=8, D=2, n_electrodes=64, n_tsteps=2000,device=torch.device("cpu")):
        super(EEGNet, self).__init__()

        self.device = device
        F2 = F1 * D
        self.T = n_tsteps  # timestamps in data
        self.C = n_electrodes    # electrodes in data

        ##################################################
        # adjust EEGNet size based on input data
        ##################################################
        # summary_window = 16 summarizes 500ms worth of EEG data
        # summary_window = 8 summarizes 250ms worth of EEG data
        # default dimension_reduction is 4

        # sets defaults
        dimension_reduction = 16
        summary_window = 16
        # use smaller filters for EEG windows <1 second
        if n_tsteps < 1000: 
            summary_window = 8
            dimension_reduction = 4

        # don't do dimensionality reduction for EEG windows <0.5 seconds
        if n_tsteps < 500:
            dimension_reduction = 1

        final_temporal_dim = (n_tsteps//30) - summary_window+1
        final_temporal_dim = (final_temporal_dim-dimension_reduction) // dimension_reduction + 1

        final_layer_size = F1*D*final_temporal_dim


        # reduce dimensionality to minimize number of parameters in model
        # dimension_reduction is 8 by default
        # for lower latency variants of EEGNet this may need to be smaller
        #dimension_reduction = 1

        self._downsample = nn.AvgPool2d((1,10)).to(self.device)  # downsamples from 1KHz to 100Hz

        # BLOCK 1 LAYERS
        #self._input_crop = crop([self.C, self.T]).to(self.device)  # takes a 2 second window of data
        
        ##################################################
        # BLOCK 1
        ##################################################
        # 500ms temperal kernals
        self._conv1 = nn.Conv2d(1, F1, (1, 50), padding='same').to(self.device)   
        self._batchnorm1 = nn.BatchNorm2d(F1, False).to(self.device)

        ##################################################
        # BLOCK 2
        ##################################################
        self._depthwise = ConstrainedDepthwiseConv2d(F1, D).to(self.device)  
        self._batchnorm2 = nn.BatchNorm2d(F1*D).to(self.device)
        # reduce sampling rate to ~32Hz
        self._avg_pool = nn.AvgPool2d((1,3)).to(self.device)
        self._dropout1 = nn.Dropout(p=0.5).to(self.device)

        # seperable convolution
        self._seperable = SeparableConv2d(F1*D, F2, (1,summary_window)).to(self.device)  # ~500ms temporal windows
        self._batchnorm3 = nn.BatchNorm2d(F2).to(self.device)
        self._dropout2 = nn.Dropout(p=0.5).to(self.device)
        
        self._reduce_dimensions = nn.AvgPool2d((1,dimension_reduction)).to(self.device)
        self._dense = ConstrainedDense(input_size=final_layer_size, output_size=output_dim).to(self.device)

        # name of EEGNet model
        self._name = "EEGNet-" + str(F1) + "," + str(D)

    def get_name(self):
        return self._name

    def forward(self, x, return_logits=False, return_dataclass=False):
        x = torch.FloatTensor(x) if isinstance(x, np.ndarray) else x
        X = self.reshape_input(x)

        # Block 1
        # -------
        X = self._conv1(X)
        X = self._batchnorm1(X)
        X = self._depthwise(X)
        X = self._batchnorm2(X)
        X = nn.ELU()(X)
        X = self._avg_pool1(X)
        X = self._dropout1(X)

        # Block 2
        # -------
        X = self._seperable(X)
        X = self._batchnorm3(X)
        X = nn.ELU()(X)
        X = self._avg_pool2(X)
        X = self._dropout2(X)
        hidden_state = torch.flatten(X, start_dim=1)           # [32, 16, 1, 1] -> [32, 16]

        # Classifier
        # ----------
        logits = self._dense(hidden_state)                          # [32, 16] -> [32, 2]
        #probs = nn.Softmax(dim=-1)(X)                   # [32, 2] -> [32, 2]
        probs = F.softmax(X, dim=-1)                     # [32, 2] -> [32, 2]
        
        if return_dataclass:
            return EEGNetOutput(probs, logits, hidden_state)
        # default: return logits? or return softmax probs?
        if not return_logits:
            return probs
        return logits
    
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
            x = torch.transpose(x, 0, 1)#.to(self.device)   # x is now (64, 100)        # use for online
            #pdb.set_trace()
            #x = x.permute(0,3,1,2).to(self.device)              # use for offline
            #x = crop(x, 0, 2000-self.T, self.C, self.T)     # (59, 2000)
            x = torch.unsqueeze(torch.unsqueeze(x, 0), 0) # (1, 1, 59, 1000)   # use for online
        else:   # x is (batch_size, 1000, 64)
            x = torch.transpose(x, 1, 2)#.to(self.device)   # x is now (batch_size, 64, 100) 
            #x = crop(x, 0, 2000-self.T, self.C, self.T)      # (batch_size, 59, 1000)
            x = torch.unsqueeze(x, 1) # (batch_size, 1, 59, 1000)   # use for online
        return x

if __name__ == '__main__':
    from torchsummary import summary
    eegnet = EEGNet(output_dim=4, n_tsteps=2000)
    summary(eegnet, input_size=(64, 250, 1))