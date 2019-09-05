import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# from models.pfp_net_new import get_pfp_net
from models.pfp_net_v6 import get_pfp_net
from matplotlib import pyplot as plt
from data import VOC_CLASSES as labels
import argparse
from lib.nms.gpu_nms import gpu_nms
from lib.nms.cpu_nms import *
import time


parser = argparse.ArgumentParser(description= 'Paralle Feature Pyramid Net Testing With Pytorch')
# train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--USE_GPU_NMS', default=True, type=bool,
                    help='USE_GPU_NMS')
parser.add_argument('--GPU_ID', default=0, type=int,
                    help='GPU_ID')
parser.add_argument('--num_class', default=2, type=int,
                    help='num_class')

args = parser.parse_args()
# 调用格式 proposals = clip_boxes(proposals, im_info[:2])
def clip_boxes(boxes, im_shape):
    """将proposals的边界限制在图片内"""
    # x1 >= 0
    boxes[0] = np.maximum(np.minimum(boxes[0], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[1] = np.maximum(np.minimum(boxes[1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[2] = np.maximum(np.minimum(boxes[2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[3] = np.maximum(np.minimum(boxes[3], im_shape[0] - 1), 0)
    return boxes

# Original NMS implementation
def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if args.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=args.GPU_ID)
    else:
        return cpu_nms(dets, thresh)

final_output_dir = './weights'
# from models.pfp_net import get_pfp_net
net = get_pfp_net('test', 512, args.num_class)    # initialize SSD
# net.load_weights('./weights/PFPNet512_VOC_final.pth')
# model = torch.load('./weights/PFPNet512_VOC_final.pth')
# model_dict = model.module.state_dict()

model_state_file = os.path.join(final_output_dir,
                                'PFPNet512_VOC_108000.pth')
# logger.info('=> loading model from {}'.format(model_state_file))
from collections import OrderedDict
state_dict = torch.load(model_state_file)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)


NMS_THRESH = 0.3
thresh = 0.4
font = cv2.FONT_HERSHEY_DUPLEX
font_clr = (255, 255, 255)
font_pt = (4, 12)
font_sc = 0.4
gpus = [int('0')]
model = torch.nn.DataParallel(net, device_ids=gpus).cuda()
# net.load_state_dict(model_dict)

# here we specify year (07 or 12) and dataset ('test', 'val', 'train')

images_path = '**/testfolder/'
result_path = '**/testfolder_result'
if not os.path.exists(result_path):
    os.makedirs(result_path)
images = os.listdir(images_path)
count = 0
for image_name in images:
    count+=1
    print(count)
    image = cv2.imread(os.path.join(images_path, image_name))
    im_shape = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    # plt.figure(figsize=(10,10))
    # plt.imshow(rgb_image)
    # plt.show()
    x = cv2.resize(image, (512, 512)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # plt.imshow(x)
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    detect_start = time.time()
    y = model(xx)
    print('INFO: Detection Time is: {}'.format(time.time() - detect_start))

    top_k=10

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    # here we ignore the background, so we start in 1
    for i in range(1, detections.size(1)):
        dets = np.zeros(detections[0, i].cpu().data.numpy().shape, dtype=np.float32)
        dets[:, 0:4] = detections[0, i].cpu().data.numpy()[:, 1:]*scale.cpu().numpy()
        dets[:, -1] = detections[0, i].cpu().data.numpy()[:, 0]
        # j = 0
        # score = detections[0,i,j,0]
        # keep = nms(detections[0, i].cpu().data.numpy(), NMS_THRESH)
        keep = nms(dets, NMS_THRESH)
        result = detections[0, i][keep, :]
        # result = detections[0, i]
        inds = np.where(result.cpu().numpy()[:, 0] >= thresh)[0]
        result = result[inds, :]
        # while detections[0,i,j,0] >= 0.4:
        if len(result)>0:
            # score = detections[0,i,j,0]
            for j in range(len(result)):
                score = result[j, 0]
                label_name = labels[1]
                display_txt = '%s: %.2f'%(label_name, score)
                # pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                pt = (result[j, 1:] * scale).cpu().numpy()
                pt = clip_boxes(pt, im_shape)
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                # color = colors[i]
                cv2.rectangle(image,(int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (0, 0, 255))
                cv2.putText(image, display_txt, (int(pt[0]), int(pt[1] - 5)), font, font_sc,
                    font_clr, lineType=16)
            cv2.imshow('image', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
            # cv2.imwrite(os.path.join(result_path,image_name),image)

