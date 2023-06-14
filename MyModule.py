from warnings import warn
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import cheb2ord
import torch
import torchaudio
from torch import nn

class filterbank(nn.Module):
    def __init__(self, fs, frequency_bands, filterStop=None, f_trans=1, gpass = 3, gstop = 30):
        super(filterbank, self).__init__()
        self.fs = fs
        self.f_trans = f_trans
        self.frequency_bands = frequency_bands
        self.filterStop = filterStop
        self.gpass = gpass
        self.gstop = gstop
        self.Nyquist_freq = self.fs / 2
        self.nFilter = len(self.frequency_bands)


    def forward(self, x):
        while (len(x.shape) < 4):
            x = x.unsqueeze(-1)
        (n_trials, n_channels, n_samples, temp) = x.size()
        all_filtered = torch.Tensor(np.zeros((n_trials, n_channels, n_samples, self.nFilter)))

        for i in range(self.nFilter):
            (l_freq, h_freq) = self.frequency_bands[i]
            f_pass = np.asarray([l_freq, h_freq])
            if self.filterStop is not None:
                f_stop = np.asarray(self.filterStop[i])
            else:
                f_stop = np.asarray([l_freq - self.f_trans, h_freq + self.f_trans])
            wp = f_pass / self.Nyquist_freq
            ws = f_stop / self.Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            data = x[:,:,:,0]

            torch_a = torch.as_tensor(a,dtype = data.dtype)
            torch_b = torch.as_tensor(b,dtype = data.dtype)
            for j in range(n_trials):
                all_filtered[j,:,:,i] = torchaudio.functional.lfilter(data[j, :, :],torch_a, torch_b)

        return all_filtered

class coherence_cropped(nn.Module):
    def __init__(self, time_length = "auto", time_stride = 1):
        super(coherence_cropped, self).__init__()
        self.time_length = time_length
        self.time_stride = time_stride

    def forward(self, x):
        while (len(x.shape) < 4):
            x = x.unsqueeze(-1)
        (n_trials, n_channels, n_samples, n_slice) = x.size()
        if self.time_length == "auto":
            self.time_length = n_samples
        n_windows_per_slice = int((n_samples - self.time_length) / self.time_stride) + 1
        y = torch.zeros(n_trials, n_channels, n_channels, n_slice * n_windows_per_slice)
        for i_trial in range(n_trials):
            for i_slice in range(n_slice):
                for i in range(n_windows_per_slice):
                    temp = x[i_trial, :, self.time_stride * i:self.time_stride * i + self.time_length, i_slice].squeeze(-1).squeeze(0)
                    y[i_trial, :, :, i_slice * n_windows_per_slice + i] = torch.corrcoef(temp)
        return y

def get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class group_temporal_filter(nn.Module):
    def __init__(self,
                 n_filterbank,
                 n_filters_time,
                 kernel_size_group,
                 stride_size = 1
                 ):
        super(group_temporal_filter, self).__init__()
        self.n_filterbank = n_filterbank
        self.n_filters_time = n_filters_time
        self.kernel_size_group = kernel_size_group if isinstance(kernel_size_group, list) else [kernel_size_group]
        self.n_group = len(self.kernel_size_group)
        self.stride_size = stride_size
        self.filter_list = nn.ModuleList([nn.Conv2d(
                    self.n_filterbank,
                    self.n_filters_time,
                    self.kernel_size_group[i],
                    stride=self.stride_size,
                    padding=(get_padding(self.kernel_size_group[i],self.stride_size),0)
                ) for i in range(self.n_group)])

        for layer in self.filter_list:
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for layer in self.filter_list:
            if 'y' in dir():
                y = torch.cat((y,layer(x)),dim = 1)
            else:
                y = layer(x)
        return y



