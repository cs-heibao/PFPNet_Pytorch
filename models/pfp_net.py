# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Junjie Zhao (zhao.jj@cidi.ai)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from math import ceil, floor
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from lib.models import RefineDetMultiBoxLoss
from layers import *
from data import voc_refinedet

from collections import OrderedDict
def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class vgg(nn.Module):

    def __init__(self):
        super(vgg, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1 = nn.Conv2d()
        self.relu1_1 = nn.ReLU(inplace=False)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1_2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2_1 = nn.ReLU(inplace=False)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2_2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3_1 = nn.ReLU(inplace=False)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3_2 = nn.ReLU(inplace=False)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3_3 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4_1 = nn.ReLU(inplace=False)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4_2 = nn.ReLU(inplace=False)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4_3 = nn.ReLU(inplace=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.feature_conv4_3 = nn.Conv2d(512,256,kernel_size=3, stride=1, padding=1, bias=False)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5_1 = nn.ReLU(inplace=False)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5_2 = nn.ReLU(inplace=False)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5_3 = nn.ReLU(inplace=False)
        # add another convs
        self.conv6_1 = nn.Conv2d(512,768,kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6_1 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.relu1_1(out)
        out = self.conv1_2(out)
        out = self.relu1_2(out)
        out = self.pool1(out)

        out = self.conv2_1(out)
        out = self.relu2_1(out)
        out = self.conv2_2(out)
        out = self.relu2_2(out)
        out = self.pool2(out)

        out = self.conv3_1(out)
        out = self.relu3_1(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)
        out = self.pool3(out)

        out = self.conv4_1(out)
        out = self.relu4_1(out)
        out = self.conv4_2(out)
        out = self.relu4_2(out)
        out = self.conv4_3(out)
        conv4 = self.relu4_3(out)
        conv4 = self.feature_conv4_3(conv4)
        out = self.relu4_3(out)
        out = self.pool4(out)

        out = self.conv5_1(out)
        out = self.relu5_1(out)
        out = self.conv5_2(out)
        out = self.relu5_2(out)
        out = self.conv5_3(out)
        out = self.relu5_3(out)

        out = self.conv6_1(out)
        out = self.relu6_1(out)
        # out = self.pool5(out)

        return conv4, out

# class SpatialPyramidPooling2d(nn.Module):
#      r"""apply spatial pyramid pooling over a 4d input(a mini-batch of 2d inputs
#  8     with additional channel dimension) as described in the paper
#  9     'Spatial Pyramid Pooling in deep convolutional Networks for visual recognition'
# 10     Args:
# 11         num_level:
# 12         pool_type: max_pool, avg_pool, Default:max_pool
# 13     By the way, the target output size is num_grid:
# 14         num_grid = 0
# 15         for i in range num_level:
# 16             num_grid += (i + 1) * (i + 1)
# 17         num_grid = num_grid * channels # channels is the channel dimension of input data
# 18     examples:
# 19         >>> input = torch.randn((1,3,32,32), dtype=torch.float32)
# 20         >>> net = torch.nn.Sequential(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1),\
# 21                                       nn.ReLU(),\
# 22                                       SpatialPyramidPooling2d(num_level=2,pool_type='avg_pool'),\
# 23                                       nn.Linear(32 * (1*1 + 2*2), 10))
# 24         >>> output = net(input)
# 25     """
#
#      def __init__(self, num_level, pool_type='max_pool'):
#          super(SpatialPyramidPooling2d, self).__init__()
#          self.num_level = num_level
#          self.pool_type = pool_type
#
#      def forward(self, x):
#          N, C, H, W = x.size()
#          for i in range(self.num_level):
#              level = i + 1
#              kernel_size = (ceil(H / level), ceil(W / level))
#              stride = (ceil(H / level), ceil(W / level))
#              padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))
#
#              if self.pool_type == 'max_pool':
#                  # tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
#                  tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding))
#              else:
#                  # tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
#                  tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding))
#
#              if i == 0:
#                  res = tensor
#              else:
#                  res = torch.cat((res, tensor), 1)
#          return res

# class SPPNet1(nn.Module):
#      def __init__(self, num_level=3, pool_type='max_pool'):
#          super(SPPNet1,self).__init__()
#          self.num_level = num_level
#          self.pool_type = pool_type
#
#          # self.num_grid = self._cal_num_grids(num_level)
#          self.spp_layer = SpatialPyramidPooling2d(num_level)
#
#      def _cal_num_grids(self, level):
#          count = 0
#          for i in range(level):
#              count += (i + 1) * (i + 1)
#          return count
#
#      def forward(self, x):
#          x = self.spp_layer(x)
#          print(x.size())
#          return x

class SppNet(nn.Module):
    def __init__(self, layers):
        super(SppNet,self).__init__()
        self.level = layers


    def forward(self, x):
        assert x.shape[0] == 1 , 'batch size need to set to be 1'
        N, C, H, W = x.size()
        self.scale = (1 / 2.) ** self.level
        spp = nn.AdaptiveMaxPool2d(output_size=(H * self.scale, W * self.scale))

        return spp
class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes):
        super(Bottleneck,self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv2d(self.inplanes,self.planes,kernel_size=1,stride=1,padding=0,bias=False)
    def forward(self, x):
        x = self.conv1(x)
        return x


class PFPNet(nn.Module):

    def __init__(self, layers, status, size):

        super(PFPNet, self).__init__()
        inplanes = 768
        planes = 256
        self.cfg = voc_refinedet
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.level = layers
        self.status = status
        self.backbone = vgg()

        self.low_dimensional0 = Bottleneck(inplanes, planes)
        self.low_dimensional1 = Bottleneck(inplanes, planes)
        self.low_dimensional2 = Bottleneck(inplanes, planes)
        self.convs = nn.Conv2d(1280,256,kernel_size=3,stride=1,padding=1,bias=False)
        self.arm_loc = nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.arm_conf = nn.Conv2d(256, 63, kernel_size=3, stride=1, padding=1, bias=False)

    def SPP(self,x,level):
        N, C, H, W = x.size()
        self.scale = (1 / 2.) ** level
        spp = nn.AdaptiveMaxPool2d(output_size=(int(H * self.scale), int(W * self.scale)))

        return spp
    def forward(self, x):


        arm_loc = list()
        arm_conf = list()


        conv4, x = self.backbone(x)
        high0 = self.SPP(x,self.level[0])
        high0 = high0(x)
        high1 = self.SPP(x,self.level[1])
        high1 = high1(x)
        high2 = self.SPP(x,self.level[2])
        high2 = high2(x)
        low0 = self.low_dimensional0(high0)
        low1 = self.low_dimensional1(high1)
        low2 = self.low_dimensional2(high2)
        # MSCA
        unsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        unsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        downsample2_0 = nn.AdaptiveAvgPool2d(output_size=(int(low0.size()[2] * 1 / 2.), int(low0.size()[3] * 1 / 2.)))
        downsample2_1 = nn.AdaptiveAvgPool2d(output_size=(int(low1.size()[2] * 1 / 2.), int(low1.size()[3] * 1 / 2.)))
        downsample4_0 = nn.AdaptiveAvgPool2d(output_size=(int(low0.size()[2] * 1 / 4.), int(low0.size()[3] * 1 / 4.)))
        # p0

        p0 = torch.cat([high0,unsample2(low1),unsample4(low2)],1)
        p0 = self.convs(p0)

        p1 = torch.cat([high1,downsample2_0(low0),unsample2(low2)],1)
        p1 = self.convs(p1)

        p2 = torch.cat([high2,downsample4_0(low0),downsample2_1(low1)],1)
        p2 = self.convs(p2)

        p = [conv4,p0, p1, p2]
        for x in p:
            arm_loc.append(self.arm_loc(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(self.arm_conf(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)

        if self.status == "test":
            pass
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 21),
                self.priors
            )

        return output




def get_pfp_net(mode,size):
    layers = [0,1,2]
    model = PFPNet(layers,mode,size)
    return model

def transform(img, size=(512,512)):
    transform1 = transforms.Compose([
        transforms.Scale(size),
        transforms.ToTensor()
    ])
    mode = transform1(img)
    return mode


def arm_multibox(vgg, extra_layers, cfg):
    arm_loc_layers = []
    arm_conf_layers = []
    vgg_source = [21, 28, -2]
    for k, v in enumerate(vgg_source):
        arm_loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * 2, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        arm_loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)
from PIL import Image
# arm_criterion = RefineDetMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5,
#                              False, args.cuda)
# layers = [0,1,2]
# model = PFPNet(layers)
# model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
# model.train()
#
#
# input = Image.open('/home/jie/project/CIDI/PFP_net/000000.jpg')
# input = transform(input).unsqueeze(0)
# input = input.type(torch.FloatTensor).cuda()
# out_put = model(input)
# arm_loss_l, arm_loss_c = arm_criterion(out_put, targets)
# print(model)