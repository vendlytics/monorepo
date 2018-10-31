# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import config, config_from_file, config_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--config', dest='config_file',
                        help='optional config file',
                        default='configs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_configs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="/srv/share/jyang375/models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo')
    parser.add_argument('--video_path', dest='video_path',
                        help='path to load videos from for demo')
    parser.add_argument('--output_string', dest='output_string',
                        help='String appended to output file')
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument(
        '--parallel_type',
        dest='parallel_type',
        help='which part of model to parallel, 0: all, 1: model before roi pooling',
        default=0,
        type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)

    args = parser.parse_args()
    return args


lr = config.TRAIN.LEARNING_RATE
momentum = config.TRAIN.MOMENTUM
weight_decay = config.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= config.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in config.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > config.TEST.MAX_SIZE:
            im_scale = float(config.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.config_file is not None:
        config_from_file(args.config_file)
    if args.set_configs is not None:
        config_from_list(args.set_configs)

    print('Using config:')
    pprint.pprint(config)
    np.random.seed(config.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception(
            'There is no input directory for loading network from ' +
            input_dir)
    load_name = os.path.join(
        input_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(
            args.checksession,
            args.checkepoch,
            args.checkpoint))

#   pascal_classes = np.asarray(['__background__',
#                        'aeroplane', 'bicycle', 'bird', 'boat',
#                        'bottle', 'bus', 'car', 'cat', 'chair',
#                        'cow', 'diningtable', 'dog', 'horse',
#                        'motorbike', 'person', 'pottedplant',
#                        'sheep', 'sofa', 'train', 'tvmonitor'])
    pascal_classes = np.asarray(['__background__',
                                 'face'])

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(
            pascal_classes,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(
            pascal_classes,
            101,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(
            pascal_classes,
            50,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(
            pascal_classes,
            152,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        config.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # pdb.set_trace()

    print("load checkpoint %s" % (load_name))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)

    if args.cuda > 0:
        config.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True

    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)
        num_images = 0
    elif args.image_dir:
        imglist = os.listdir(args.image_dir)
        num_images = len(imglist)

        print('Loaded Photo: {} images.'.format(num_images))
    elif args.video_path:
        out_dir = 'output/video'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        dets_file_name = os.path.join(
            out_dir, 'video-det-fold-%s.txt' %
            args.output_string)
        fid = open(dets_file_name, 'w')

        print(args.video_path)
        if not os.path.exists(args.video_path):
            print('Video does not exist.')

        video = cv2.VideoCapture(args.video_path)

        num_images = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get width, height
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

        # Define the codec and create VideoWriter object
        # TODO: The videos I am using are 30fps, but you should
        # programmatically get this.
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(
            'output/video/output-%s.avi' %
            args.output_string, fourcc, 30.0, (width, height))

    processed_images = 0

    while (num_images >= 0):
        processed_images += 1

        total_tic = time.time()
        if webcam_num == -1 and not args.video_path:
            num_images -= 1

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError(
                    "Webcam could not open. Please check connection.")
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        elif args.image_dir:
            im_file = os.path.join(args.image_dir, imglist[processed_images])
            # im = cv2.imread(im_file)
            im_in = np.array(imread(im_file))
        elif args.video_path:
            ret, frame = video.read()
            im_in = np.array(frame)
            if not ret:
                break

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        # rgb -> bgr
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        # pdb.set_trace()
        det_tic = time.time()

        # actual prediction
        rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if config.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if config.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(config.TRAIN.BBOX_NORMALIZE_STDS).cuda(
                    ) + torch.FloatTensor(config.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(config.TRAIN.BBOX_NORMALIZE_STDS).cuda(
                    ) + torch.FloatTensor(config.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(
                        1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im2show = np.copy(im)
        for j in xrange(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, config.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    # TODO: want: frame_number x_min y_min x_max y_max confidence_score
                    # cls_dets.cpu().numpy()'s last float is confidence score!

                    # print('dets: ', cls_dets.cpu().numpy())

                    # writeout
                    for detection in cls_dets.cpu().numpy():
                        fid.write(
                            str(processed_images) +
                            "\t" +
                            "\t".join(map(str, detection)) + "\n")

                    im2show = vis_detections(
                        im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' .format(
                processed_images + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

        if vis and args.image_dir:
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)
            result_path = os.path.join(
                args.image_dir, imglist[processed_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)
        elif vis and args.video_path:
            out.write(im2show)
        else:
            im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", im2showRGB)
            # print('frame: ', im2showRGB)
            total_toc = time.time()
            total_time = total_toc - total_tic
            frame_rate = 1 / total_time
            # print('Frame rate:', frame_rate)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if webcam_num >= 0:
        cap.release()
        cv2.destroyAllWindows()

fid.close()
