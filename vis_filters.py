# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Hangyan Jiang, based on code from Ross Girshick
# --------------------------------------------------------


"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from train import combined_roidb
from lib.utils.test import test_net

import matplotlib.pyplot as plt

CLASSES = ('__background__',
           'tampered')

# PLEASE specify weight files dir for vgg16
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',), 'res50': ('resnetv1_faster_rcnn_iter_10000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res50]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

def get_model():
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('default', 'Columbia', 'default', NETS[demonet][0])

    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res50':
        net = resnetv1(batch_size=1, num_layers=50)
    else:
        raise NotImplementedError
    #net.create_architecture(sess, "TEST", 2, tag='default', anchor_scales=[8, 16, 32])

    srm2 = tf.get_variable('vgg_16/srm2/weights', [5, 5, 3, 3], trainable=False)
    srm3 = tf.get_variable('vgg_16/srm3/weights', [5, 5, 3, 3], trainable=False)
    #srm2 = tf.get_variable('resnet_v1_50/srm/weights', [5, 5, 3, 3], trainable=False)

    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    #variables = tf.global_variables()
    #for i in variables:
    #    print(i)

    filters = sess.run(srm2)
    print(filters.shape)

    plt.imsave('filt0.jpg', cv2.resize(filters[:, :, :, 0], (800, 800), interpolation=cv2.INTER_NEAREST))
    plt.imsave('filt1.jpg', cv2.resize(filters[:, :, :, 1], (800, 800), interpolation=cv2.INTER_NEAREST))
    plt.imsave('filt2.jpg', cv2.resize(filters[:, :, :, 2], (800, 800), interpolation=cv2.INTER_NEAREST))

    filters2 = sess.run(srm3)
    print(filters2.shape)

    plt.imsave('filt3.jpg', cv2.resize(filters2[:, :, :, 0], (800, 800), interpolation=cv2.INTER_NEAREST))
    plt.imsave('filt4.jpg', cv2.resize(filters2[:, :, :, 1], (800, 800), interpolation=cv2.INTER_NEAREST))
    plt.imsave('filt5.jpg', cv2.resize(filters2[:, :, :, 2], (800, 800), interpolation=cv2.INTER_NEAREST))


    #print('Loaded network {:s}'.format(tfmodel))

    #imdb, _ = combined_roidb("Columbia")


    #test_net(sess, net, imdb, weights_filename='output')

if __name__ == '__main__':
    get_model()
