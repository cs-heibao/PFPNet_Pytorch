# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by cs-heibao 2019.02
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
