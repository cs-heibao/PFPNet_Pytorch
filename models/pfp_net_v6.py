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
from data import *
import copy
import os

from collections import OrderedDict
def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

BN_MOMENTUM = 0.1
class PFPNet(nn.Module):

    def __init__(self, layers, status, size, classes, backbone_, spp_, bottleneck_, head_):

        super(PFPNet, self).__init__()
        inplanes = 768
        planes = 256
        self.cfg = pfp
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.level = layers
        self.status = status
        self.num_classes = classes

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 6)
        self.vgg = nn.ModuleList(backbone_)
        self.spp = nn.ModuleList(spp_)
        self.bottleneck = nn.ModuleList(bottleneck_)
        self.pfp_conf = nn.ModuleList(head_[0])
        self.pfp_loc = nn.ModuleList(head_[1])

        self.convs_4 = nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(768, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.convs = nn.Conv2d(1536, 256,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        # self.arm_loc = nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1, bias=False)
        # self.arm_conf = nn.Conv2d(256, self.num_classes*3, kernel_size=3, stride=1, padding=1, bias=False)
        if status == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, size, 0, 200, 0.01, 0.45)

    def forward(self, x):

        sources = list()
        arm_loc = list()
        arm_conf = list()
        feature_h = list()
        feature_l = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        # s = self.L2Norm(x)
        x = self.L2Norm(x)
        conv4 = self.convs_4(x)
        conv4 = self.bn1(conv4)
        conv4 = self.relu(conv4)
        # conv4 = self.L2Norm(conv4)


        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        # feature_h.append(conv4)

        # apply spp
        for k,v in enumerate(self.spp):
            if k==0:
                feature_h.append(v(conv4))
            else:
                feature_h.append(v(x))

        # apply bottleneck
        for k, v in enumerate(self.bottleneck):
            feature_l.append(self.bn2(v(feature_h[k])))
        # apply MSCA
        for i in range(len(feature_h)):
            if i==0:
                k = i+1
                temp = []
                for j in range(k,len(feature_h)):
                    unsample = nn.UpsamplingBilinear2d(scale_factor=2**j)
                    temp.append(unsample(feature_l[j]))
                temp.insert(i,feature_h[i])
                sources.append(self.bn2(self.convs(torch.cat(temp, 1))))
            elif i ==len(feature_h)-1:
                k = i + 1
                temp = []
                for j in range(0, i):
                    scale = (1 / 2.) ** (i - j)
                    downsample = nn.AdaptiveAvgPool2d(output_size=(int(feature_l[j].size()[2] * scale), int(feature_l[j].size()[3] * scale)))
                    temp.append(downsample(feature_l[j]))
                    # unsample = nn.UpsamplingBilinear2d(scale_factor=2 ** j)
                    # temp.append(unsample(feature_l[j]))
                temp.insert(i, feature_h[i])
                sources.append(self.bn2(self.convs(torch.cat(temp, 1))))
            else:
                k = i+1
                temp = []
                for j in range(i):
                    scale = (1/2.)**(i-j)
                    downsample = nn.AdaptiveAvgPool2d(output_size=(int(feature_l[j].size()[2]* scale), int(feature_l[j].size()[3]* scale)))
                    temp.append(downsample(feature_l[j]))
                for j in range(k, len(feature_h)):
                    scale = 2**(j-i)
                    unsample = nn.UpsamplingBilinear2d(scale_factor=scale)
                    temp.append(unsample(feature_l[j]))
                temp.insert(i, feature_h[i])
                sources.append(self.bn2(self.convs(torch.cat(temp, 1))))

        for (x, l, c) in zip(sources, self.pfp_loc, self.pfp_conf):
        # for x in sources:
            # self.convs = nn.Conv2d(x.shape[1], 256, kernel_size=3, stride=1, padding=1, bias=False)
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)

        if self.status == "test":
            output = self.detect(
                arm_loc.view(arm_loc.size(0), -1, 4),  # loc preds
                self.softmax(arm_conf.view(arm_conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output


base = {
    '512': [{'conv1':[64, 64, 'M'], 'conv2':[128, 128, 'M'], 'conv3':[256, 256, 256, 'C'], 'conv4':[512, 512, 512, 'M'],
            'conv5':[512, 512, 512]}]
}

extra_spp = {'512': [0, 0, 1, 2]}
bottle = {'512': [[0, 1, 2, 3], [768, 768, 768, 768], [256]]}
mbox = {'512': [4, 6, 6, 4, 256]# number of boxes per feature map location
}

def vgg(infos, i, batch_norm=False):
    layers = []
    in_channels = i
    vgg_names = sorted(list(infos[0]))
    for name in vgg_names:
        cfg = infos[0][name]
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                if name == 'conv5':
                    rate = 2
                    ker = 3
                    pad = int(rate*(ker-1)/2)
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=pad, dilation=2)
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 768, kernel_size=3, padding=3, dilation=3)
    layers += [pool5, conv6, nn.BatchNorm2d(768, momentum=BN_MOMENTUM),
               nn.ReLU(inplace=True)]
    return layers

def spp(cfg):
    layers = []
    for i in cfg:
        layers += [nn.MaxPool2d(kernel_size=2**i, stride=2**i)]
    return layers

def bottleneck(cfg):
    layers = []
    for i in range(len(cfg[0])):
        layers+= [nn.Conv2d(cfg[1][i], cfg[2][0], kernel_size=1, stride=1, padding=0, bias=False)]
    return layers

def multibox(vgg,spp,bottleneck, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    for k in range(len(cfg)-1):
        loc_layers += [nn.Conv2d(cfg[-1], cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(cfg[-1], cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, spp, bottleneck, (conf_layers, loc_layers)

def get_pfp_net(mode, size, classes):
    layers = [0,1,2]
    backbone_, spp_, bottleneck_ , head_= multibox(vgg(base[str(size)],3), spp(extra_spp[str(size)]),
                                            bottleneck(bottle[str(size)]), mbox[str(size)], classes)

    model = PFPNet(layers, mode, size, classes, backbone_, spp_, bottleneck_, head_)
    return model
