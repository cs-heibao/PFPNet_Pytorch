# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by JunjieAI 2019.02
# ------------------------------------------------------------------------------
# import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.optim as optim
import torch.utils.data as data
import torch.nn.init as init
import torch.nn as nn
import torchvision.transforms as transform
from tensorboardX import SummaryWriter
import argparse
import os
import sys
from utils.logging import Logger
from data import *
import time
import datetime
from lib.models import RefineDetMultiBoxLoss
from utils.augmentations import SSDAugmentation
# from models.pfp_net_new import get_pfp_net
from models.pfp_net_v6 import get_pfp_net
# from models.pfp_net_new_box18 import get_pfp_net


parser = argparse.ArgumentParser(description= 'Paralle Feature Pyramid Net Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', type=str,
                    help='VOC')
parser.add_argument('--input_size', default='512', type=str,
                    help='PFP512')
parser.add_argument('--num_class', default=2, type=int,
                    help='num_class')
parser.add_argument('--dataset_root', default='RootPath',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='./pretained/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--mode',default='train',type=str,
                    help='exchange the mode of this scripts')
parser.add_argument('--resume', default='./weights', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

sys.stdout = Logger(os.path.join(args.save_folder, 'log.txt'))

def adjust_learning_rate(optimizer, gamma, step):

    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    if args.dataset == 'COCO':
        '''if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))'''
    elif args.dataset == 'VOC':
        '''if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')'''
        cfg = pfp[args.input_size]
        dataset = VOCDetection(root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))

    refinedet_net = get_pfp_net(args.mode, args.input_size, args.num_class)
    net = refinedet_net
    # print(net)

    if args.cuda:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        f_list = os.listdir(args.resume)
        name = []
        for i in f_list:
            if os.path.splitext(i)[1] == '.pth':
                name.append(os.path.splitext(i)[0].split('_')[-1])
        if len(name)==0:
            pass
        else:
            idx = max(list(map(int, name)))
            args.start_iter = idx + 1
            args.resume = os.path.join(args.resume,'PFPNet512_VOC_{}.pth'.format(idx))
            print('Resuming training, loading {}...'.format(args.resume))
            from collections import OrderedDict
            state_dict = torch.load(args.resume)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # name = k[7:]  # remove `module.`
                name = k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)

            # net.load_weights(args.resume)
    else:
        pretrained_dict = torch.load(args.basenet)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # filter out unnecessary keys
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)


    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    arm_criterion = RefineDetMultiBoxLoss(args.num_class, 0.5, True, 0, True, 3, 0.5,
                                          False, args.cuda)

    net.train()
    print('Loading the dataset...')

    print('Training PFPNet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    log_time = datetime.datetime.now()

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    # batch_iterator = enumerate(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, iteration)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        else:
            images = images
            targets = [ann for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = arm_criterion(out, targets)
        loss = loss_l + loss_c
        try:
            loss.backward()
        except Exception as e:
            print('something is wrong')
            continue
        optimizer.step()
        t1 = time.time()
        with open('./{}_{}_{}_{}:{}:{}loss.txt'.format(log_time.year, log_time.month, log_time.day,log_time.hour,
                                                       '%02d'%log_time.minute, '%02d'%log_time.second),'a') as fd:
            fd.write(str(loss) + '\n')


        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('INFO: ' +
                'iter ' + repr(iteration) + ' || Localization Loss: %.4f Confidence Loss: %.4f Total Loss: %.4f||' \
                % (loss_l.item(), loss_c.item(), loss.item()), end=' ')

        if iteration != 0 and iteration % 2000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), args.save_folder
                       + '/PFPNet{}_{}_{}.pth'.format(args.input_size, args.dataset,
                                                         repr(iteration)))
    torch.save(net.state_dict(), args.save_folder
               + '/PFPNet{}_{}_final.pth'.format(args.input_size, args.dataset))


if __name__=='__main__':

    train()
