from __future__ import print_function
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import SingleConv3D
import math
import numpy as np

class PRNet(t.nn.Module):
    def __init__(self,  inc=1, patch_size=1, n_classes=5, base_chns=12, droprate=0, norm='in', depth = False, dilation=1):
        super(PRNet, self).__init__()
        self.model_name = "seg"
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')  # 1/4(h,h)
        self.downsample = nn.MaxPool3d(2, 2)  # 1/2(h,h)
        self.drop = nn.Dropout(droprate)

        self.conv0_1 = SingleConv3D(inc, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv0_2 = SingleConv3D(base_chns, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv1_1 = SingleConv3D(base_chns, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv1_2 = SingleConv3D(base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv2_1 = SingleConv3D(2*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv2_2 = SingleConv3D(2 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv3_1 = SingleConv3D(4*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv3_2 = SingleConv3D(4 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv4_1 = SingleConv3D(8*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')
        self.conv4_2 = SingleConv3D(8 * base_chns, 16 * base_chns, norm=norm, depth=depth, dilat=math.ceil(dilation/2), pad='same')

        self.conv5_1 = SingleConv3D(24*base_chns, 8*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv5_2 = SingleConv3D(8 * base_chns, 8 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv6_1 = SingleConv3D(12*base_chns, 4*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv6_2 = SingleConv3D(4 * base_chns, 4 * base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv7_1 = SingleConv3D(6*base_chns, 2*base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv7_2 = SingleConv3D(2 * base_chns,  base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.conv8_1 = SingleConv3D(2*base_chns, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')
        self.conv8_2 = SingleConv3D( base_chns, base_chns, norm=norm, depth=depth, dilat=dilation, pad='same')

        self.classification = nn.Sequential(
            nn.Conv3d(in_channels=base_chns, out_channels=n_classes, kernel_size=1),
        )
        fc_inc = int(np.asarray(patch_size).prod()/4096)*16*base_chns
        self.fc1 = nn.Linear(fc_inc, 8 * base_chns)
        self.fc2 = nn.Linear(8 * base_chns, 4 * base_chns)
        self.fc3 = nn.Linear(4 * base_chns, 3)

    def forward(self, x, out_feature=True):
        out = self.conv0_1(x)
        conv0 = self.conv0_2(out)
        out = self.downsample(conv0)
        out = self.conv1_1(out)
        conv1 = self.conv1_2(out)
        out = self.downsample(conv1)  # 1/2
        out = self.conv2_1(out)
        conv2 = self.conv2_2(out)  #
        out = self.downsample(conv2)  # 1/4
        out = self.conv3_1(out)
        conv3 = self.conv3_2(out)  #
        out = self.downsample(conv3)  # 1/8
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        #out = self.drop(out)

        fc_out = out.view(out.shape[0],-1)
        fc_out = self.fc1(fc_out)
        fc_out = self.fc2(fc_out)
        fc_out = self.fc3(fc_out)

        up5 = self.upsample(out)  # 1/4
        out = t.cat((up5, conv3), 1)
        out = self.conv5_1(out)
        conv5 = self.conv5_2(out)
        center_feature5 = conv5[:, :, conv5.shape[2]//2, conv5.shape[3]//2,conv5.shape[4]//2]

        up6 = self.upsample(conv5)  # 1/2
        out = t.cat((up6, conv2), 1)
        out = self.conv6_1(out)
        conv6 = self.conv6_2(out)
        center_feature6 = conv6[:, :, conv6.shape[2]//2, conv6.shape[3]//2, conv6.shape[4]//2]

        up7 = self.upsample(out)
        out = t.cat((up7, conv1), 1)
        out = self.conv7_1(out)
        conv7 = self.conv7_2(out)
        center_feature7 = conv7[:, :, conv7.shape[2]//2, conv7.shape[3]//2, conv7.shape[4]//2]

        up8 = self.upsample(conv7)
        out = t.cat((up8, conv0), 1)
        out = self.conv8_1(out)
        conv8 = self.conv8_2(out)
        center_feature8 = conv8[:, :, conv8.shape[2]//2, conv8.shape[3]//2, conv8.shape[4]//2]

        out = self.classification(out)
        dic = {'fc_position': fc_out, 'ae': out}
        if out_feature:
            dic['center_feature5'] = center_feature5
            dic['center_feature6'] = center_feature6
            dic['center_feature7'] = center_feature7
            dic['center_feature8'] = center_feature8
            dic['fine_feature'] = conv8
        return dic