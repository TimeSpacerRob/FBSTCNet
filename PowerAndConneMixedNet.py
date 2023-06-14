# This is the PyTorch implementation of the FBSTCNet-M architecture for EEG-based emotion classification.

# Reference:
# "W. Huang, W. Wang, Y. Li, W. Wu. FBSTCNet: A Spatio-Temporal Convolutional Network Integrating Power and Connectivity Features for EEG-Based Emotion Decoding. 2023. (under review)"
# 
# Email: huangwch96@gmail.com

import torch
import numpy as np
from torch import nn
from torch.nn import init

from braindecode.util import np_to_th
from braindecode.models.modules import Expression, Ensure4d
from braindecode.models.functions import (
    safe_log, square, transpose_time_to_spat
)
from MyModule import coherence_cropped, filterbank
from MyFunctions import coherence, squeeze_final_output, shift_3rd_dim_output


def get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class PowerAndConneMixedNet(nn.Module):


    def __init__(
        self,
        in_chans,
        n_classes,
        fs=100,
        f_trans = 2,
        filterRange=None,
        filterStop =None,
        input_window_samples=None,
        n_filters_time=72,
        filter_time_length=25,
        n_filters_spat=72,
        n_filters_power=36,
        pool_time_length=80,
        pool_time_stride=5,
        final_conv_length=35,
        final_conv_stride=25,
        conn_nonlin=coherence,
        pool_mode="mean",
        pool_nonlin=safe_log,
        split_first_layer=True,
        batch_norm=True,
        same_filters_for_features = True,
        batch_norm_alpha=0.1,
        drop_prob=0.5,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.final_conv_stride = final_conv_stride
        self.same_filters_for_features = same_filters_for_features
        if(self.same_filters_for_features):
            self.n_filters_power = self.n_filters_spat
            self.n_filters_coherence = self.n_filters_spat
        else:
            self.n_filters_power = n_filters_power
            self.n_filters_coherence = self.n_filters_spat-self.n_filters_power

        self.conn_nonlin = conn_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.filterRange = filterRange
        self.n_filterbank = len(self.filterRange)
        self.fs = fs
        self.f_trans =f_trans
        self.filterStop = filterStop

        # filter bank
        if self.filterRange is not None:
            self.add_module("filterbank", filterbank(fs=self.fs,frequency_bands=self.filterRange,filterStop=self.filterStop,f_trans=self.f_trans))
        else:
            self.add_module("filterbank", Ensure4d())  # [numSample × numChannel × numPoint × 1]
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        padding_size = get_padding((self.filter_time_length, 1))

        self.add_module("dimshuffle", Expression(transpose_time_to_spat)) # [numSample × 1 × numPoint × numChannel]
        
        # temporal convolution
        self.add_module(
            "conv_time",
            nn.Conv2d(
                self.n_filterbank,
                self.n_filters_time,
                (self.filter_time_length, 1),
                stride=1,
                padding=(padding_size,0),
            ),
        )
        self.add_module("conv_nonlin_exp", Expression(square))
        self.add_module(
            "poolfunc",
            pool_class(
                kernel_size=(pool_time_length, 1),
                stride=(pool_time_stride, 1),
            ),
        )
        
        # spatial convolutions for power and connectivity-based network, respectively
        self.add_module(
            "conv_spat_power",
            nn.Conv2d(
                self.n_filters_power,
                self.n_filters_power,
                (1, self.in_chans),
                stride=1,
                groups=self.n_filters_power,
                bias=not self.batch_norm,
            ),
        )
        self.add_module(
            "conv_spat_conne",
            nn.Conv2d(
                self.n_filters_coherence,
                self.n_filters_coherence,
                (1, self.in_chans),
                stride=1,
                groups=self.n_filters_coherence,
                bias=not self.batch_norm,
            ),
        )
        if self.batch_norm:
            self.add_module(
                "bnorm_power",
                nn.BatchNorm2d(
                    self.n_filters_power, momentum=self.batch_norm_alpha, affine=True
                ),
            )
            self.add_module(
                "bnorm_conne",
                nn.BatchNorm2d(
                    self.n_filters_coherence, momentum=self.batch_norm_alpha, affine=True
                ),
            )
        #power-based feature extraction
        self.add_module("pool_nonlin_exp", Expression(safe_log))
        
        #connectivity-based feature extraction
        self.connectivity_exp = coherence_cropped(time_length=200, time_stride=100)
        self.add_module("power_drop", nn.Dropout(p=self.drop_prob))
        self.add_module("conne_drop", nn.Dropout(p=self.drop_prob))
        
        #final convolution
        self.add_module(
            "conv_power_classifier",
            nn.Conv2d(
                self.n_filters_power,
                self.n_classes,
                (self.final_conv_length, 1),
                stride=(self.final_conv_stride, 1),
                bias=True,
            ),
        )
        self.add_module(
            "conv_conn_classifier",
            nn.Conv2d(
                self.n_filters_coherence,
                self.n_classes,
                (self.n_filters_coherence, 1),
                bias=True,
            ),
        )
        self.add_module("conne_shift", Expression(shift_3rd_dim_output))

        self.add_module("softmax", nn.LogSoftmax(dim=1))
        self.add_module("squeeze", Expression(squeeze_final_output))

        
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.constant_(self.conv_time.bias, 0)
        init.xavier_uniform_(self.conv_conn_classifier.weight, gain=1)
        init.constant_(self.conv_conn_classifier.bias, 0)
        init.xavier_uniform_(self.conv_power_classifier.weight, gain=1)
        init.constant_(self.conv_power_classifier.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm_power.weight, 1)
            init.constant_(self.bnorm_power.bias, 0)
            init.constant_(self.bnorm_conne.weight, 1)
            init.constant_(self.bnorm_conne.bias, 0)
        init.xavier_uniform_(self.conv_spat_conne.weight, gain=1)
        if not self.batch_norm:
            init.constant_(self.conv_spat_conne.bias, 0)
        init.xavier_uniform_(self.conv_spat_power.weight, gain=1)
        if not self.batch_norm:
            init.constant_(self.conv_spat_power.bias, 0)

    def forward(self, x):

        x = self.filterbank(x)
        x = self.dimshuffle(x)
        x = self.conv_time(x)
        if self.n_filters_power > 0:
            x1 = x[:, 0:self.n_filters_power, :, :]
            x1 = self.conv_spat_power(x1)
            if self.batch_norm:
                x1 = self.bnorm_power(x1)
            x1 = self.conv_nonlin_exp(x1)
            x1 = self.poolfunc(x1)
            x1 = self.pool_nonlin_exp(x1)
            x1 = self.power_drop(x1)
            x1 = self.conv_power_classifier(x1)

        if self.n_filters_coherence > 1:
            if self.same_filters_for_features:
                x2 = x[:, 0:self.n_filters_coherence, :, :]
            else:
                x2 = x[:, self.n_filters_power:self.n_filters_power+self.n_filters_coherence, :, :]
            x2 = self.conv_spat_conne(x2)
            if self.batch_norm:
                x2 = self.bnorm_conne(x2)
            x2 = self.connectivity_exp(x2)
            x2 = self.conv_conn_classifier(x2)
            x2 = self.conne_shift(x2)

        if self.n_filters_power > 0:
            if self.n_filters_coherence > 1:
                xout = torch.cat((x1, x2), dim=2)
            else:
                xout = x1
        else:
            xout = x2
        xout = self.softmax(xout)
        xout = self.squeeze(xout)

        return xout





