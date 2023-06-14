import torch
import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord

def coherence(x):
    if x.dim() == 4:
        y = torch.zeros(x.size()[0],x.size()[1],x.size()[1],x.size()[3])
        for i in range(0,x.size()[0]):
            for j in range(0,x.size()[3]):
                temp = x[i,:,:,j].squeeze(-1).squeeze(0)
                y[i,:,:,j] = torch.corrcoef(temp)

    return y




def squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    return x

def squeeze_final_output_2d(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def squeeze_3rd_dim_output(x):
    assert x.size()[2] == 1
    x = torch.squeeze(x,2)
    return x

def shift_3rd_dim_output(x):
    assert x.size()[2] == 1
    x = torch.transpose(x,2,3)
    return x




