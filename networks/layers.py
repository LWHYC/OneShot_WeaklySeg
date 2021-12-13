#coding:utf8
'''
常用的层,比如inception block,residual block
'''
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

class depthwise(nn.Module):
    '''
    depthwise convlution
    '''
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=3, dilation=1, depth=False):
        super(depthwise, self).__init__()
        if depth:
                self.Conv=nn.Sequential(OrderedDict([('conv1_1_depth', nn.Conv3d(cin, cin,
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=cin)),
                        ('conv1_1_point', nn.Conv3d(cin, cout, 1))]))
        else:
            if stride>=1:
                self.Conv=nn.Conv3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
            else:
                stride = int(1//stride)
                self.Conv = nn.ConvTranspose3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
    def forward(self, x):
        return self.Conv(x)

class parallel_depthwise(nn.Module):
    '''
    depthwise convlution, with 3 direction parralleled
    '''
    def __init__(self, cin, cout, kernel_size, stride, padding, dilation, depth=False):
        super(parallel_depthwise, self).__init__()
        if depth:
                self.Conv=nn.Sequential(OrderedDict([('conv1_1_depth', nn.Conv3d(cin, cin,
                        kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=cin)),
                        ('conv1_1_point', nn.Conv3d(cin, cout, 1))]))
        else:
            if stride>=1:
                self.Ax_conv=nn.Conv3d(cin, cout, kernel_size=kernel_size['ax'], stride=stride,
                                                            padding=padding['ax'], dilation=dilation['ax'])
                self.Co_conv=nn.Conv3d(cin, cout, kernel_size=kernel_size['co'], stride=stride,
                                                            padding=padding['co'], dilation=dilation['co'])
                self.Sa_conv=nn.Conv3d(cin, cout, kernel_size=kernel_size['sa'], stride=stride,
                                                            padding=padding['sa'], dilation=dilation['sa'])
            else:
                stride = int(1//stride)
                self.Conv = nn.ConvTranspose3d(cin, cout, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, dilation=dilation)
    def forward(self, x):
        return self.Ax_conv(x)+self.Co_conv(x)+self.Sa_conv(x)


class Deconv(nn.Module):
    def __init__(self, cin, cout):
        super(Deconv, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose2d(cin, cout, 2, stride=2)),
            ('norm', nn.BatchNorm2d(cout)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)

class Deconv3D(nn.Module):
    def __init__(self, cin, cout, norm='in'):

        super(Deconv3D, self).__init__()
        if norm == 'bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.model = nn.Sequential(OrderedDict([
            ('deconv1', nn.ConvTranspose3d(cin, cout, 2, stride=2)),
            ('norm', Norm(cout)),
            ('relu', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.model(x)

class SingleConv(nn.Module):
    def __init__(self, cin, cout, padding=1):
        super(SingleConv, self).__init__()
        self.padding = padding
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(cin, cout, 3, padding=self.padding)),
            ('norm1_1', nn.BatchNorm2d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)

class SingleConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', pad=1, depth=False, dilat=1):
        super(SingleConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        if pad =='same':
            self.padding = dilat
        else:
            self.padding = pad
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, 3, padding=self.padding, depth=depth, dilation=dilat)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.model(x)

class SingleResConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', pad=1, depth=False, dilat=1):
        super(SingleResConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        self.dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.norm = Norm(cout)
        self.active = nn.ReLU()
        if pad =='same':
            self.padding = dilat
        else:
            self.padding = pad
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, 3, padding=self.padding, depth=depth, dilation=dilat)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):

        return self.active(self.norm(self.model(x)+self.Input(x)))

class SingleSeparateConv3D(nn.Module):
    def __init__(self, cin, cout, stride=1, norm='in', pad=1, depth=False, dilat=1):
        super(SingleSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        self.padding = [[0, 1, 1], [1, 0, 0]]
        self.conv = [[1, 3, 3], [3, 1, 1]]
        self.dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        if pad == 'same':
            self.padding = [[0, dilat, dilat], [dilat, 0, 0]]
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, self.conv[1], stride=stride, padding=self.padding[1], depth=depth, dilation=self.dilation[1])),
            ('conv1_2', depthwise(cout, cout, self.conv[0], stride=stride, padding=self.padding[0], depth=depth, dilation=self.dilation[0])),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.model(x)

class SingleResSeparateConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', pad=1, depth=False, dilat=1):
        super(SingleResSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        self.padding = [[0, 1, 1], [1, 0, 0]]
        self.conv = [[1, 3, 3], [3, 1, 1]]
        self.dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.norm = Norm(cout)
        self.active = nn.ReLU()
        if pad == 'same':
            self.padding = [[0, dilat, dilat], [dilat, 0, 0]]
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, self.conv[1], padding=self.padding[1], depth=depth, dilation=self.dilation[1])),
            ('conv1_2', depthwise(cout, cout, self.conv[0], padding=self.padding[0], depth=depth, dilation=self.dilation[0])),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):

        return self.active(self.norm(self.model(x)+self.Input(x)))

class DenseSingleConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', padding=1, depth=False, dilat=1):
        super(DenseSingleConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        if padding =='same':
            self.padding = dilat
        else:
            self.padding = padding
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, 3, padding=self.padding, depth=depth, dilation=dilat)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):
        return torch.cat(self.model(x), x)


class DenseSingleSeparateConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', pad=1, depth=False, dilat=1):
        super(DenseSingleSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        self.padding = [[0, 1, 1], [1, 0, 0]]
        self.conv = [[1, 3, 3], [3, 1, 1]]
        self.dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        if pad == 'same':
            self.padding = [[0, dilat, dilat], [dilat, 0, 0]]
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, self.conv[1], padding=self.padding[1], depth=depth, dilation=self.dilation[1])),
            ('conv1_2', depthwise(cout, cout, self.conv[0], padding=self.padding[0], depth=depth, dilation=self.dilation[0])),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):
        return torch.cat((self.model(x), x),1)

class SingleConvSE3D(nn.Module):
    def __init__(self, cin, cout, norm='in', pad=1, depth=False, dilat=1):
        super(SingleConvSE3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        if pad =='same':
            self.padding = dilat
        else:
            self.padding = pad
        self.model = nn.Sequential(OrderedDict([
            ('SE', SEBlock(cin, cout)),
            ('conv1_1', depthwise(cin, cout, 3, padding=self.padding, depth=depth, dilation=dilat)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.model(x)

class DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm.1', nn.BatchNorm2d(input_num, momentum=0.0003)),

        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm.2', nn.BatchNorm2d(num1, momentum=0.0003)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseAsppBlock, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature

class DenseAsppBlock3D(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, norm='in',norm_start=True):
        super(DenseAsppBlock3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        if norm_start:
            self.add_module('norm1', Norm(input_num, momentum=0.0003)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', Norm(num1, momentum=0.0003)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseAsppBlock3D, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature

class DenseAsppSeperateBlock3D(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, norm='in',norm_start=True):
        super(DenseAsppSeperateBlock3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        if norm_start:
            self.add_module('norm1', Norm(input_num, momentum=0.0003)),

        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', SingleResSeparateConv3D(cin=input_num, cout=num1, dilat=dilation_rate, norm=norm, pad='same')),

        self.add_module('norm2', Norm(num1, momentum=0.0003)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', SingleResSeparateConv3D(cin=num1, cout=num2, dilat=dilation_rate, norm=norm, pad='same')),

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseAsppSeperateBlock3D, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature



class Transition(nn.Module):
    def __init__(self, cin, cout, norm='in', depth=False):
        super(Transition, self).__init__()
        if norm == 'bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            raise ValueError('please choose the correct normilze method!!!')
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', depthwise(cin, cout, 1, depth=depth, padding=0)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.model(x)


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(cin, cout, 3, padding=1)),
            ('norm1_1', nn.BatchNorm2d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(cout, cout, 3, padding=1)),
            ('norm1_2', nn.BatchNorm2d(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.model(x)

class DoubleConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', droprate=0, depth=False):
        super(DoubleConv3D, self).__init__()
        if norm == 'bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(cout, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU()),
        ]))
        self.norm = Norm(cout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.model(x)
        out = self.norm(out)
        return self.activation(out)

class DoubleResConv3D(nn.Module):
    def __init__(self, cin, cout, norm='in', droprate=0, depth=False):
        super(DoubleResConv3D, self).__init__()
        if norm == 'bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(cout, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU()),
        ]))
        self.norm = Norm(cout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class DoubleResSeparateConv3D(nn.Module):
    '''
    在xyz方向上分开卷积，xy方向上卷3次，z方向上1次
    '''
    def __init__(self, cin, cout, kernel_size=3,norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True, separate_direction='axial'):
        super(DoubleResSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()

        if separate_direction == 'axial':
            if pad == 'same':
                padding = [[0, dilat, dilat], [dilat, 0, 0]]
            else:
                padding = [[0, 1, 1], [1, 0, 0]]
            conv = [[1, kernel_size, kernel_size], [kernel_size, 1, 1]]
            dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        elif separate_direction == 'sagittal':
            if pad == 'same':
                padding = [[dilat, 0, dilat], [0, dilat, 0]]
            else:
                padding = [[1, 0, 1], [0, 1, 0]]
            conv = [[kernel_size, 1, kernel_size], [1, kernel_size, 1]]
            dilation = [[dilat, 1, dilat], [1, dilat, 1]]
        elif separate_direction == 'coronal':
            if pad == 'same':
                padding = [[dilat, dilat, 0], [0, 0, dilat]]
            else:
                padding = [[1, 1, 0], [0, 0, 1]]
            conv = [[kernel_size, kernel_size, 1], [1, 1, kernel_size]]
            dilation = [[dilat, dilat, 1], [1, 1, dilat]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_1_depth_deep',
             depthwise(cout, cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(cout)),
            ('relu1_1_deep', nn.ReLU(inplace=True)),
        ]))

        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class DoubleResSeparateConv25D(nn.Module):
    '''
    在xyz方向上分开卷积，xy方向上卷3次，z方向上1次
    '''
    def __init__(self, cin, cout, kernel_size=3,norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True, separate_direction='axial'):
        super(DoubleResSeparateConv25D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()

        if separate_direction == 'axial':
            if pad == 'same':
                padding = [[0, dilat, dilat], [dilat, 0, 0]]
            else:
                padding = [[0, 1, 1], [1, 0, 0]]
            conv = [[1, kernel_size, kernel_size], [kernel_size, 1, 1]]
            dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        elif separate_direction == 'sagittal':
            if pad == 'same':
                padding = [[dilat, 0, dilat], [0, dilat, 0]]
            else:
                padding = [[1, 0, 1], [0, 1, 0]]
            conv = [[kernel_size, 1, kernel_size], [1, kernel_size, 1]]
            dilation = [[dilat, 1, dilat], [1, dilat, 1]]
        elif separate_direction == 'coronal':
            if pad == 'same':
                padding = [[dilat, dilat, 0], [0, 0, dilat]]
            else:
                padding = [[1, 1, 0], [0, 0, 1]]
            conv = [[kernel_size, kernel_size, 1], [1, 1, kernel_size]]
            dilation = [[dilat, dilat, 1], [1, 1, dilat]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
        ]))

        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class TribleResConv(nn.Module):

    def __init__(self, cin, cout):
        super(TribleResConv, self).__init__()
        self.Input = nn.Conv2d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(cin, cout, 3, padding=3, dilation=3)),
            ('norm1_1', nn.BatchNorm2d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_1', nn.Conv2d(cout, cout, 3, padding=3, dilation=3)),
            ('norm1_1', nn.BatchNorm2d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(cout, cout, 3, padding=3, dilation=3)),
            ('norm1_2', nn.BatchNorm2d(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
        ]))
        self.norm = nn.BatchNorm2d(cout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class TribleResConv3D(nn.Module):

    def __init__(self, cin, cout, norm='in', droprate=0, depth=False):
        super(TribleResConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(cout, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_3', nn.ReLU()),
            ('conv1_3_depth', depthwise(cout, cout, 3, padding=1, depth=depth, dilation=1)),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU()),
        ]))
        self.norm = Norm(cout)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class TriSeparateConv3D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3,norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True, separate_direction='axial'):
        super(TriSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        if separate_direction == 'axial':
            if pad == 'same':
                padding = [[0, dilat, dilat], [dilat, 0, 0]]
            else:
                padding = [[0, 1, 1], [1, 0, 0]]
            conv = [[1, kernel_size, kernel_size], [kernel_size, 1, 1]]
            dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        elif separate_direction == 'sagittal':
            if pad == 'same':
                padding = [[dilat, 0, dilat], [0, dilat, 0]]
            else:
                padding = [[1, 0, 1], [ 0, 1, 0]]
            conv = [[kernel_size, 1, kernel_size], [1, kernel_size, 1]]
            dilation = [[dilat, 1, dilat], [1, dilat, 1]]
        elif separate_direction == 'coronal':
            if pad == 'same':
                padding = [[dilat, dilat, 0], [0, 0, dilat]]
            else:
                padding = [[1, 1, 0], [0, 0, 1]]
            conv = [[kernel_size,kernel_size , 1], [1, 1, kernel_size]]
            dilation = [[dilat, dilat, 1], [1, 1, dilat]]

        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_1_depth_deep', depthwise(cout, cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(cout)),
            ('relu1_1_deep', nn.ReLU(inplace=True)),
            ('conv1_2_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU(inplace=True)),
        ]))

        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x)
        out = self.norm(out)
        return self.activation(out)

class TriResSeparateConv25D(nn.Module):
    '''
    在xyz方向上分开卷积，xy方向上卷3次，z方向上1次
    '''
    def __init__(self, cin, cout, kernel_size=3,norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True, separate_direction='axial'):
        super(TriResSeparateConv25D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()

        if separate_direction == 'axial':
            if pad == 'same':
                padding = [[0, dilat, dilat], [dilat, 0, 0]]
            else:
                padding = [[0, 1, 1], [1, 0, 0]]
            conv = [[1, kernel_size, kernel_size], [kernel_size, 1, 1]]
            dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        elif separate_direction == 'sagittal':
            if pad == 'same':
                padding = [[dilat, 0, dilat], [0, dilat, 0]]
            else:
                padding = [[1, 0, 1], [0, 1, 0]]
            conv = [[kernel_size, 1, kernel_size], [1, kernel_size, 1]]
            dilation = [[dilat, 1, dilat], [1, dilat, 1]]
        elif separate_direction == 'coronal':
            if pad == 'same':
                padding = [[dilat, dilat, 0], [0, 0, dilat]]
            else:
                padding = [[1, 1, 0], [0, 0, 1]]
            conv = [[kernel_size, kernel_size, 1], [1, 1, kernel_size]]
            dilation = [[dilat, dilat, 1], [1, 1, dilat]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU(inplace=True)),
        ]))

        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class TriResSeparateConv3D(nn.Module):
    '''
    在xyz方向上分开卷积，xy方向上卷3次，z方向上1次
    '''
    def __init__(self, cin, cout, kernel_size=3,norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True, separate_direction='axial'):
        super(TriResSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()

        if separate_direction == 'axial':
            if pad == 'same':
                padding = [[0, dilat, dilat], [dilat, 0, 0]]
            else:
                padding = [[0, 1, 1], [1, 0, 0]]
            conv = [[1, kernel_size, kernel_size], [kernel_size, 1, 1]]
            dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        elif separate_direction == 'sagittal':
            if pad == 'same':
                padding = [[dilat, 0, dilat], [0, dilat, 0]]
            else:
                padding = [[1, 0, 1], [0, 1, 0]]
            conv = [[kernel_size, 1, kernel_size], [1, kernel_size, 1]]
            dilation = [[dilat, 1, dilat], [1, dilat, 1]]
        elif separate_direction == 'coronal':
            if pad == 'same':
                padding = [[dilat, dilat, 0], [0, 0, dilat]]
            else:
                padding = [[1, 1, 0], [0, 0, 1]]
            conv = [[kernel_size, kernel_size, 1], [1, 1, kernel_size]]
            dilation = [[dilat, dilat, 1], [1, 1, dilat]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_1_depth_deep', depthwise(cout, cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(cout)),
            ('relu1_1_deep', nn.ReLU(inplace=True)),
            ('conv1_2_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU(inplace=True)),
        ]))

        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class TriResParallelSeparateConv3D(nn.Module):
    '''
    在xyz方向上分开卷积，xy方向上卷3次，z方向上1次
    '''
    def __init__(self, cin, cout, kernel_size=3,norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True, separate_direction='axial'):
        super(TriResParallelSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        padding=[{},{}]
        conv=[{},{}]
        dilation=[{},{}]
        if pad == 'same':
            padding[0]['ax'] = [0, dilat, dilat]
            padding[0]['sa'] = [dilat, 0, dilat]
            padding[0]['co'] = [dilat, dilat, 0]
            padding[1]['ax'] = [dilat, 0, 0]
            padding[1]['sa'] = [0, dilat, 0]
            padding[1]['co'] = [0, 0, dilat]
        else:
            padding[0]['ax'] = [0, 1, 1]
            padding[0]['sa'] = [1, 0, 1]
            padding[0]['co'] = [1, 1, 0]
            padding[1]['ax'] = [1, 0, 0]
            padding[1]['sa'] = [0, 1, 0]
            padding[1]['co'] = [0, 0, 1]
        conv[0]['ax']= [1, kernel_size, kernel_size]
        conv[0]['sa'] = [kernel_size, 1, kernel_size]
        conv[0]['co'] = [kernel_size, kernel_size, 1]
        conv[1]['ax'] = [kernel_size, 1, 1]
        conv[1]['sa'] = [1, kernel_size, 1]
        conv[1]['co'] = [1, 1, kernel_size]
        dilation[0]['ax'] = [1, dilat, dilat]
        dilation[0]['sa'] = [dilat, 1, dilat]
        dilation[0]['co'] = [dilat, dilat, 1]
        dilation[1]['ax'] = [dilat, 1, 1]
        dilation[1]['sa'] =  [1, dilat, 1]
        dilation[1]['co'] = [1, 1, dilat]

        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', parallel_depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, stride=1, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_1_depth_deep', parallel_depthwise(cout, cout, conv[1], padding=padding[1], depth=depth, stride=1, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(cout)),
            ('relu1_1_deep', nn.ReLU(inplace=True)),
            ('conv1_2_depth', parallel_depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, stride=1, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3_depth', parallel_depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, stride=1, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU(inplace=True)),
        ]))

        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class SETriResSeparateConv3D(nn.Module):
    '''
    在xyz方向上分开卷积，xy方向上卷3次，z方向上1次
    '''
    def __init__(self, cin, cout, norm='in', droprate=0, depth=False, pad=0, dilat=1):
        super(SETriResSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()

        padding = [[0, 1, 1], [1, 0, 0]]
        conv = [[1, 3, 3], [3, 1, 1]]
        dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        if pad == 'same':
            padding = [[0, dilat, dilat], [dilat, 0, 0]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.SElayer = SELayer3D(cout)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_1_depth_deep', depthwise(cout, cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(cout)),
            ('relu1_1_deep', nn.ReLU()),
            ('conv1_2_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU()),
            ('conv1_3_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU()),
        ]))
        self.norm = Norm(cout)
        self.activation = nn.ReLU()

    def forward(self, x):
        input = self.Input(x)
        out = self.model(x)+input+self.SElayer(input)
        out = self.norm(out)
        out = out
        return self.activation(out)

class InceptTriResSeparateConv3D(nn.Module):
    '''
    在xyz方向上分开卷积，xy方向上卷3次，z方向上1次
    '''
    def __init__(self, cin, cout, norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True):
        super(InceptTriResSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()

        incet_cout=round(cout/2)
        padding = [[0, 1, 1], [1, 0, 0]]
        conv = [[1, 3, 3], [3, 1, 1]]
        dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        if pad == 'same':
            padding = [[0, dilat, dilat], [dilat, 0, 0]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.Incept1 = nn.Sequential(OrderedDict([
            ('conv1_0_depth', depthwise(cin, incet_cout, 1, padding=0, depth=depth, dilation=1)),
            ('drop1_0', nn.Dropout(droprate)),
            ('norm1_0', Norm(incet_cout)),
            ('relu1_0', nn.ReLU()),
            ('conv1_1_depth', depthwise(incet_cout, incet_cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(incet_cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_1_depth_deep', depthwise(incet_cout, incet_cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(incet_cout)),
            ('relu1_1_deep', nn.ReLU()),
            ('conv1_2_depth', depthwise(incet_cout, incet_cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(incet_cout)),
            ('relu1_2', nn.ReLU()),
            ('conv1_3_depth', depthwise(incet_cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU()),
        ]))

        self.Incept2 = nn.Sequential(OrderedDict([
            ('conv1_0_depth', depthwise(cin, incet_cout, 1, padding=0, depth=depth, dilation=1)),
            ('drop1_0', nn.Dropout(droprate)),
            ('norm1_0', Norm(incet_cout)),
            ('relu1_0', nn.ReLU()),
            ('conv1_1_depth', depthwise(incet_cout, incet_cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(incet_cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_1_depth_deep', depthwise(incet_cout, incet_cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(incet_cout)),
            ('relu1_1_deep', nn.ReLU()),
            ('conv1_2_depth', depthwise(incet_cout, incet_cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(incet_cout)),
            ('relu1_2', nn.ReLU()),
            ('conv1_3_depth', depthwise(incet_cout, incet_cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU()),
            ('conv1_1_depth',
             depthwise(incet_cout, incet_cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(incet_cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_1_depth_deep',
             depthwise(incet_cout, incet_cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(incet_cout)),
            ('relu1_1_deep', nn.ReLU()),
            ('conv1_2_depth',
             depthwise(incet_cout, incet_cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(incet_cout)),
            ('relu1_2', nn.ReLU()),
            ('conv1_3_depth',
             depthwise(incet_cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU()),
        ]))


        self.active = nn.ReLU()
        self.norm = Norm(cout)
    def forward(self, x):
        out = self.Incept1(x)+self.Incept2(x)+self.Input(x)
        out = self.norm(out)
        return self.active(out)


class QuadResSeparateConv3D(nn.Module):
    '''
    在xyz方向上分开卷积，xy方向上卷3次，z方向上1次
    '''
    def __init__(self, cin, cout, kernel_size=3,norm='in', droprate=0, depth=False, pad=0, dilat=1, active=True, separate_direction='axial'):
        super(QuadResSeparateConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()

        if separate_direction == 'axial':
            if pad == 'same':
                padding = [[0, dilat, dilat], [dilat, 0, 0]]
            else:
                padding = [[0, 1, 1], [1, 0, 0]]
            conv = [[1, kernel_size, kernel_size], [kernel_size, 1, 1]]
            dilation = [[1, dilat, dilat], [dilat, 1, 1]]
        elif separate_direction == 'sagittal':
            if pad == 'same':
                padding = [[dilat, 0, dilat], [0, dilat, 0]]
            else:
                padding = [[1, 0, 1], [0, 1, 0]]
            conv = [[kernel_size, 1, kernel_size], [1, kernel_size, 1]]
            dilation = [[dilat, 1, dilat], [1, dilat, 1]]
        elif separate_direction == 'coronal':
            if pad == 'same':
                padding = [[dilat, dilat, 0], [0, 0, dilat]]
            else:
                padding = [[1, 1, 0], [0, 0, 1]]
            conv = [[kernel_size, kernel_size, 1], [1, 1, kernel_size]]
            dilation = [[dilat, dilat, 1], [1, 1, dilat]]
        self.Input = nn.Conv3d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1_depth', depthwise(cin, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_1', nn.Dropout(droprate)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_1_depth_deep', depthwise(cout, cout, conv[1], padding=padding[1], depth=depth, dilation=dilation[1])),
            ('drop1_1_deep', nn.Dropout(droprate)),
            ('norm1_1_deep', Norm(cout)),
            ('relu1_1_deep', nn.ReLU(inplace=True)),
            ('conv1_2_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_2', nn.Dropout(droprate)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_3', nn.Dropout(droprate)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU(inplace=True)),
            ('conv1_4_depth', depthwise(cout, cout, conv[0], padding=padding[0], depth=depth, dilation=dilation[0])),
            ('drop1_4', nn.Dropout(droprate)),
            ('norm1_4', Norm(cout)),
            ('relu1_4', nn.ReLU(inplace=True)),
        ]))

        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.model(x)+self.Input(x)
        out = self.norm(out)
        return self.activation(out)

class DLAConv(nn.Module):

    def __init__(self, cin, cout):
        super(DLAConv, self).__init__()
        self.Input = nn.Conv2d(cin, cout, 1)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(cin, cout, 3, padding=3, dilation=3)),
            ('norm1_1', nn.BatchNorm2d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_1', nn.Conv2d(cin, cout, 3, padding=3, dilation=3)),
            ('norm1_1', nn.BatchNorm2d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(cout, cout, 3, padding=3, dilation=3)),
            ('norm1_2', nn.BatchNorm2d(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=2)),
        ]))
        self.norm = nn.BatchNorm2d(cout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        Add = self.model(x)+self.Input(x)
        Norm = self.norm(Add)
        return  self.activation(Norm)

class DLAConv3D(nn.Module):

    def __init__(self, cin, cout, norm='in'):
        super(DLAConv3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.Input = nn.Conv3d(cin, cout, 1, stride=2)
        self.model = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, cout, 3, padding=3, dilation=3)),
            ('norm1_1', Norm(cout)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv3d(cin, cout, 3, padding=3, dilation=3)),
            ('norm1_2', Norm(cout)),
            ('relu1_2', nn.ReLU()),
            ('conv1_3', nn.Conv3d(cout, cout, 3, padding=3, dilation=3)),
            ('norm1_3', Norm(cout)),
            ('relu1_3', nn.ReLU()),
            ('pool', nn.MaxPool3d(kernel_size=2)),
        ]))
        self.norm = Norm(cout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.model(x) + self.Input(x)
        out = self.norm(out)
        return  self.activation(out)


class AggreResNode(nn.Module):

    def __init__(self, cin, cout):
        super(AggreResNode, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(cin, cout, 3, padding=3, dilation=3)),
            ('norm1', nn.BatchNorm2d(cout))
        ]))
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.activation = nn.ReLU()

    def forward(self, ida, x):
        x = self.Upsample(x)
        input = torch.cat((ida, x), 1)
        out = x+self.model(input)
        return self.activation(out)

class AggreResNode3D(nn.Module):

    def __init__(self, cin, cout, norm='in',depth=False):
        super(AggreResNode3D, self).__init__()
        if norm =='bn':
            Norm = nn.BatchNorm3d
        elif norm == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', depthwise(cin, cin, kernel_size=3, padding=3, dilation=3, depth=depth)),
            ('norm1', Norm(cout))
        ]))
        self.Upsample = nn.Upsample(scale_factor=2)
        self.activation = nn.ReLU()

    def forward(self, ida, x):
        x = self.Upsample(x)
        input = torch.cat((ida, x), 1)
        out = x+self.model(input)
        return self.activation(out)


class Inception_v1(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception_v1, self).__init__()
        assert (co % 4 == 0)
        cos = [int(co / 4)] * 4
        cin = int(cin)
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm2d(co))
        if relu: self.activa.add_module('relu', nn.ReLU())

        self.branch1 = nn.Conv2d(cin, cos[0], 1, stride=2)

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(cin, 2 * cos[1], 1)),
            ('norm1_1', nn.BatchNorm2d(2 * cos[1])),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv2d(2 * cos[1], 2 * cos[1], kernel_size=(1, 3) , stride=(0, 2), padding=(0, 1))),
            ('norm1_2', nn.BatchNorm2d(2 * cos[1])),
            ('relu1_2', nn.ReLU()),
            ('conv1_2', nn.Conv2d(2 * cos[1], 2 * cos[1], kernel_size=(3, 1) , stride=(2, 0), padding=(1, 0)))
        ]))

        self.branch3 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv2d(cin, 2 * cos[2], 1, stride=1)),
            ('norm2_1', nn.BatchNorm2d(2 * cos[2])),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv2d(2 * cos[2], cos[2], 5, stride=2, padding=2)),
        ]))

        self.branch4 = nn.Sequential(OrderedDict([
            ('pool3_1', nn.MaxPool2d(2)),
            ('conv3_1', nn.Conv2d(cin, cos[3], 1, stride=1))
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = torch.cat((branch1, branch2, branch3, branch4), 1)
        return self.activa(result)

class Inception_v1_3D(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True, stride=2, normmethod='in', depth=False):
        super(Inception_v1_3D, self).__init__()
        if normmethod =='bn':
            Norm = nn.BatchNorm3d
        elif normmethod == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        assert (co % 12 == 0)
        cos = [int(co / 4)] * 4
        cin = int(cin)
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', Norm(co))
        if relu: self.activa.add_module('relu', nn.ReLU())

        self.branch1 = nn.Conv3d(cin, cos[0], 1, stride=stride)

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, 2 * cos[1], 1)),
            ('norm1_1', Norm(co)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(2 * cos[1], cos[1], kernel_size=3, stride=stride, padding=1, depth=depth)),
        ]))

        self.branch3 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv3d(cin, 2 * cos[2], 1, stride=1)),
            ('norm2_1', Norm(co)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2_depth', depthwise(2 * cos[2], cos[2], kernel_size=5, stride=stride, padding=2, depth=depth)),  # depthwise convolution
        ]))

        self.branch4 = nn.Sequential(OrderedDict([
            ('pool3', nn.MaxPool3d(2, stride=stride)),
            ('conv3_1', nn.Conv3d(cin, cos[3], 1))
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = torch.cat((branch1, branch2, branch3, branch4), 1)
        return self.activa(result)

class Inception_v2(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception_v2, self).__init__()
        assert (co % 4 == 0)
        cos = [int(co / 4)] * 4
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm2d(co))
        if relu: self.activa.add_module('relu', nn.ReLU(True))

        self.branch1 = nn.Conv2d(cin, cos[0], 1)

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(cin, 2 * cos[1], 1)),
            ('norm1_1', nn.BatchNorm2d(2 * cos[1])),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(2 * cos[1], cos[1], 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv2d(cin, 2 * cos[2], 1, stride=1)),
            ('norm2_1', nn.BatchNorm2d(2 * cos[2])),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(2 * cos[2], cos[2], 5, stride=1, padding=2)),
        ]))

        self.branch4 = nn.Sequential(OrderedDict([
            ('pool3_1', nn.MaxPool3d(3, stride=1, padding=1)),
            ('conv3_2', nn.Conv3d(cin, cos[3], 1, stride=1))
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = torch.cat((branch1, branch2, branch3, branch4), 1)
        return self.activa(result)

class Inception_v2_3D(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True, normmethod = 'in',depth = False):
        super(Inception_v2_3D, self).__init__()
        if normmethod == 'bn':
            Norm = nn.BatchNorm3d
        elif normmethod == 'in':
            Norm = nn.InstanceNorm3d
        else:
            print('please choose the correct normilze method!!!')
            os._exit()
        assert (co % 4 == 0)
        cos = [int(co / 4)] * 4
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', Norm(co))
        if relu: self.activa.add_module('relu', nn.ReLU())

        self.branch1 = nn.Conv3d(cin, cos[0], 1)

        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin, 2 * cos[1], 1)),
            ('norm1_1', Norm(2*cos[1])),
            ('relu1_1', nn.ReLU()),
            ('conv1_2_depth', depthwise(2 * cos[1], cos[1], 3, stride=1, padding=1, depth=depth)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv3d(cin, 2 * cos[2], 1, stride=1)),
            ('norm2_1', Norm(2*cos[2])),
            ('relu2_1', nn.ReLU()),
            ('conv2_2_depth', depthwise(2 * cos[2], cos[2], 5, stride=1, padding=2, depth=depth)),
        ]))

        self.branch4 = nn.Sequential(OrderedDict([
            ('pool3', nn.MaxPool3d(3, stride=1, padding=1)),
            ('conv3_1', nn.Conv3d(cin, cos[3], 1, stride=1))
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = torch.cat((branch1, branch2, branch3, branch4), 1)
        return self.activa(result)

class res_conc_block(nn.Module):
    # '''
    # 残差链接模块
    # 分支1：3*3，stride=1的卷积
    # 分支2:1*1，stride=1的卷积，3*3，stride=1的卷积
    # 分支3：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=1的卷积
    # 分支1,2,3concat到一起，1*1，stride=1卷积
    # 最后在与input相加
    # '''
    def __init__(self, cin, cn, norm=True, relu=True):
        super(res_conc_block, self).__init__()
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm2d(3 * cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Conv2d(cin, cn, 3, padding=1)
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(cin, cn, 1)),
            ('norm1', nn.BatchNorm2d(cn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(cn, cn, 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(cin, cn, 1)),
            ('norm1', nn.BatchNorm2d(cn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(cn, cn, 3, stride=1, padding=1)),
            ('norm2', nn.BatchNorm2d(cn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(cn, cn, 3, stride=1, padding=1)),
        ]))
        self.merge = nn.Conv2d(3 * cn, cin, 1, 1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        result = torch.cat((branch1, branch2, branch3), 1)
        result = self.activa(result)
        return x + self.merge(result)

class Res_concv_block(nn.Module):
    # '''
    # 残差链接模块
    # input: 1*1,stride=1的卷积
    # 分支1：3*3，stride=1的卷积
    # 分支2:1*1，stride=1的卷积，3*3，stride=1的卷积
    # 分支3：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=1的卷积
    # 分支1,2,3concat到一起，1*1，stride=1卷积
    # 最后在与input相加
    # '''
    def __init__(self, cin, cn, norm=True, relu=True):
        super(Res_concv_block, self).__init__()
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm2d(3 * cn))
        if relu: self.activa.add_module('relu', nn.ReLU(True))
        self.input = nn.Conv2d(cin, cn, 1)
        self.branch1 = nn.Conv2d(cin, cn, 3, padding=1)
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(cin, cn, 1)),
            ('norm1', nn.BatchNorm2d(cn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(cn, cn, 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(cin, cn, 1)),
            ('norm1', nn.BatchNorm2d(cn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(cn, cn, 3, stride=1, padding=1)),
            ('norm2', nn.BatchNorm2d(cn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(cn, cn, 3, stride=1, padding=1)),
        ]))
        self.merge = nn.Conv2d(3 * cn, cn, 1, 1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        result = torch.cat((branch1, branch2, branch3), 1)
        result = self.activa(result)
        return self.input(x) + self.merge(result)

class SEResBlock3D(nn.Module):
    '''
    SEResBlock,注意力机制
    '''
    def __init__(self, cin, cout, stride=1, norm='in', pad='same', dilation=1, ratio=4):
        super(SEResBlock3D, self).__init__()
        self.singleconv = SingleSeparateConv3D(cout, cout, stride=stride, norm=norm, pad=pad, depth=False,
                                               dilat=dilation)
        self.SElayer = SELayer3D(cout, cout)
        if stride >= 1:
            self.Input = nn.Conv3d(cin, cout, 1, stride=stride, dilation=dilation, padding=0)
        else:
            stride = int(1//stride)
            self.Input = nn.ConvTranspose3d(cin, cout, 1, stride=stride, dilation=dilation, padding=0)
        self.actv = nn.ReLU()

    def forward(self, x):
        x = self.Input(x)
        out = self.singleconv(x)
        out = self.SElayer(out)
        out = out+x
        out = self.actv(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEBlock, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channels
        self.acti_func1 = nn.ReLU(inplace=True)
        self.acti_func2 = nn.Sigmoid()

        self.pool1 = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(self.in_chns, self.out_chns, 1)
        self.fc2 = nn.Conv3d(self.out_chns, self.in_chns, 1)

    def forward(self, x):
        f = self.pool1(x)
        f = self.fc1(f)
        f = self.acti_func1(f)
        f = self.fc2(f)
        f = self.acti_func2(f)
        return f * x + x


class SELayer3D(nn.Module):
    def __init__(self, channel):
        super(SELayer3D, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, round(channel/2), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(round(channel/2), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)