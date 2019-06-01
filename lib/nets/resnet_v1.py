# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import numpy as np

from lib.nets.network import Network
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from lib.config import config as cfg
from lib.utils.compact_bilinear_pooling import compact_bilinear_pooling_layer

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.FLAGS.weight_decay,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  batch_norm_params = {
    # NOTE 'is_training' here does not work because inside resnet it gets reset:
    # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
    'is_training': False,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'trainable': cfg.FLAGS.bn_train,
    'updates_collections': ops.GraphKeys.UPDATE_OPS
  }

  with arg_scope(
      [slim.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      trainable=is_training,
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc

class resnetv1(Network):
  def __init__(self, batch_size=1, num_layers=50):
    Network.__init__(self, batch_size=batch_size)
    self._num_layers = num_layers
    self._resnet_scope = 'resnet_v1_%d' % num_layers

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bboxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be backpropagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], 1))
      if cfg.FLAGS.MAX_POOL:
        pre_pool_size = cfg.FLAGS.POOLING_SIZE * 2
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                         name="crops")
        crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
      else:
        crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.FLAGS.POOLING_SIZE, cfg.FLAGS.POOLING_SIZE],
                                         name="crops")
    return crops

  # Do the first few layers manually, because 'SAME' padding can behave inconsistently
  # for images of different sizes: sometimes 0, sometimes 1
  def build_base(self, ver=''):
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1'+ver)
      net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
      net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1'+ver)

    return net

  def build_network(self, sess, is_training=True, ver=''):
    # select initializers
    if cfg.FLAGS.initializer:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)
    bottleneck = resnet_v1.bottleneck
    # choose different blocks for different number of layers

    def resnet_v1_block(scope, bottleneck, base_depth, num_units, stride):
      return resnet_utils.Block(scope, bottleneck, [{
          'depth': base_depth * 4,
          'depth_bottleneck': base_depth,
          'stride': 1
      }] * (num_units - 1) + [{
          'depth': base_depth * 4,
          'depth_bottleneck': base_depth,
          'stride': stride
      }])

    if self._num_layers == 50:
        blocks = [resnet_v1_block('block1'+ver, bottleneck, base_depth=64, num_units=3, stride=2),
                       resnet_v1_block('block2'+ver, bottleneck, base_depth=128, num_units=4, stride=2),
                       # use stride 1 for the last conv4 layer
                       resnet_v1_block('block3'+ver, bottleneck, base_depth=256, num_units=6, stride=1),
                       resnet_v1_block('block4'+ver, bottleneck, base_depth=512, num_units=3, stride=1),
        ]

    elif self._num_layers == 101:
      blocks = [
        resnet_utils.Block('block1'+ver, bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2'+ver, bottleneck,
                           [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3'+ver, bottleneck,
                           [(1024, 256, 1)] * 22 + [(1024, 256, 1)]),
        resnet_utils.Block('block4'+ver, bottleneck, [(2048, 512, 1)] * 3)
      ]
    elif self._num_layers == 152:
      blocks = [
        resnet_utils.Block('block1'+ver, bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        resnet_utils.Block('block2'+ver, bottleneck,
                           [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        # Use stride-1 for the last conv4 layer
        resnet_utils.Block('block3'+ver, bottleneck,
                           [(1024, 256, 1)] * 35 + [(1024, 256, 1)]),
        resnet_utils.Block('block4'+ver, bottleneck, [(2048, 512, 1)] * 3)
      ]
    else:
      # other numbers are not supported
      raise NotImplementedError

    assert (0 <= cfg.FLAGS.fixed_blocks < 4)
    if cfg.FLAGS.fixed_blocks == 3:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:cfg.FLAGS.fixed_blocks],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope)
        net_noise = self.build_base(ver='n')
        net_conv4_noise, _ = resnet_v1.resnet_v1(net_noise,
                                           blocks[0:cfg.FLAGS.fixed_blocks],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope, reuse=True)

    elif cfg.FLAGS.fixed_blocks > 0:
      with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net = self.build_base()
        net, _ = resnet_v1.resnet_v1(net,
                                     blocks[0:cfg.FLAGS.fixed_blocks],
                                     global_pool=False,
                                     include_root_block=False,
                                     scope=self._resnet_scope, reuse=tf.AUTO_REUSE)

        net_noise = self.build_base(ver='n')
        net_noise, _ = resnet_v1.resnet_v1(net_noise,
                                            blocks[0:cfg.FLAGS.fixed_blocks],
                                            global_pool=False,
                                            include_root_block=False,
                                            scope=self._resnet_scope, reuse=tf.AUTO_REUSE)

      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[cfg.FLAGS.fixed_blocks:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope, reuse=tf.AUTO_REUSE)
        net_conv4_noise, _ = resnet_v1.resnet_v1(net_noise,
                                            blocks[0:cfg.FLAGS.fixed_blocks],
                                            global_pool=False,
                                            include_root_block=False,
                                            scope=self._resnet_scope, reuse=tf.AUTO_REUSE)
    else:  # cfg.RESNET.FIXED_BLOCKS == 0
      with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        net = self.build_base()
        net_conv4, _ = resnet_v1.resnet_v1(net,
                                           blocks[0:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope, reuse=tf.AUTO_REUSE)
        net_noise = self.build_base(ver='n')
        net_conv4_noise, _ = resnet_v1.resnet_v1(net_noise,
                                           blocks[0:cfg.FLAGS.fixed_blocks],
                                           global_pool=False,
                                           include_root_block=False,
                                           scope=self._resnet_scope, reuse=tf.AUTO_REUSE)

    self._act_summaries.append(net_conv4)
    self._layers['head'] = net_conv4
    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # build the anchors for the image
      self._anchor_component()

      # rpn
      rpn = slim.conv2d(net_conv4, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
      self._act_summaries.append(rpn)
      rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_cls_score')
      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
      rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
      rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
      rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
      if is_training:
        rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
        # Try to have a determinestic order for the computing graph, for reproducibility
        with tf.control_dependencies([rpn_labels]):
          rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
      else:
        if cfg.FLAGS.test_mode == 'nms':
          rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        elif cfg.FLAGS.test_mode == 'top':
          rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
        else:
          raise NotImplementedError

      # rcnn
      if cfg.FLAGS.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv4, rois, "pool5")
        pool5_forNoise = self._crop_pool_layer(net_conv4_noise, rois, "pool5n")
        # Compact Bilinear Pooling
        cbp = compact_bilinear_pooling_layer(pool5, pool5_forNoise, 1024)
        #cbp_flat = slim.flatten(cbp, scope='cbp_flatten')
      else:
        raise NotImplementedError

    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
      fc7 = resnet_v1.resnet_v1(cbp,
                                   blocks[-1:],
                                   global_pool=False,
                                   include_root_block=False,
                                   scope=self._resnet_scope)

    with tf.variable_scope(self._resnet_scope, self._resnet_scope):
      # Average pooling done by reduce_mean
      fc7 = tf.reduce_mean(fc7, axis=[1, 2])
      fc7_flat = slim.flatten(fc7, scope='fc7_flatten')
      cls_score = slim.fully_connected(fc7_flat, self._num_classes, weights_initializer=initializer,
                                       trainable=is_training, activation_fn=None, scope='cls_score')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
      bbox_pred = slim.fully_connected(fc7_flat, self._num_classes * 4, weights_initializer=initializer_bbox,
                                       trainable=is_training,
                                       activation_fn=None, scope='bbox_pred')
    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["cls_score"] = cls_score
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred
    self._predictions["rois"] = rois

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

  def get_variables_to_restore(self, variables, var_keep_dic, sess, pretrained_model):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._resnet_scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic:
        print('Varibles restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix Resnet V1 layers..')
    with tf.variable_scope('Fix_Resnet_V1') as scope:
      with tf.device("/cpu:0"):
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._resnet_scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._resnet_scope + '/conv1/weights:0'],
                           tf.reverse(conv1_rgb, [2])))
