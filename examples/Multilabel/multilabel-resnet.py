#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: multilabel-resnet.py
# Author: Xiaosong Wang <xswang82@gmail.com>

import cv2
import sys
import argparse
import numpy as np
import os
import multiprocessing

import tensorflow as tf
from tensorpack.tfutils.sessinit import get_model_loader
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.dataflow.image import AugmentImageComponent
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

TOTAL_BATCH_SIZE = 12 #* 4
INPUT_SHAPE = 512
DEPTH = None
LABEL_SIZE = 14
STEPS_EPOCH = 5000
MAX_EPOCH = 100


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None, LABEL_SIZE], 'multilabel')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = tf.cast(image, tf.float32) * (1.0 / 255)

        # Wrong mean/std are used for compatibility with pre-trained models.
        # Should actually add a RGB-BGR conversion here.
        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        image = tf.transpose(image, [0, 3, 1, 2])

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = cfg[DEPTH]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', block_func, 128, defs[1], 2)
                      .apply(layer, 'group2', block_func, 256, defs[2], 2)
                      .apply(layer, 'group3', block_func, 512, defs[3], 2)
                      .BNReLU('bnlast')
                      #.Conv2D('conv_CAM', 1024, 3, stride=1, nl=BNReLU)  # transition layer added to resnet (w/o group = 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linearML', 14, nl=tf.identity)())
        costs = []
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        xentropy = multilabel_class_balanced_sigmoid_cross_entropy(
            logits=logits, label=label, name='xentropy')
        costs.append(xentropy)
        pred = tf.cast(tf.greater(logits, 0.5), tf.int32, name='prediction')
        wrong = tf.cast(tf.not_equal(pred, label), tf.float32)
        wrong = tf.reduce_mean(wrong, name='train_error')
        # add_moving_summary(tf.reduce_mean(wrong, name='train-error'))
        # # wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        # # add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        # #
        # # wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        # # add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        #
        # wd_cost = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
        # add_moving_summary(loss, wd_cost)
        # self.cost = tf.add_n([loss, wd_cost], name='cost')

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            costs.append(wd_cost)

            add_param_summary(('.*/W', ['histogram']))   # monitor W
            self.cost = tf.add_n(costs, name='cost')
            add_moving_summary(costs + [wrong, self.cost])

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.001, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(name):
    # return FakeData([[64, 224,224,3],[64]], 1000, random=False, dtype='uint8')
    isTrain = name == 'train'

    # datadir = args.data
    # ds = dataset.ILSVRC12(datadir, train_or_test,
    #                       shuffle=True if isTrain else False, dir_structure='original')
    ds = dataset.CHESTXRAY14(name, shuffle=True)
    if isTrain:
        # class Resize(imgaug.ImageAugmentor):
        #     """
        #     crop 8%~100% of the original image
        #     See `Going Deeper with Convolutions` by Google.
        #     """
        #     def _augment(self, img, _):
        #         h, w = img.shape[:2]
        #         area = h * w
        #         for _ in range(10):
        #             targetArea = self.rng.uniform(0.08, 1.0) * area
        #             aspectR = self.rng.uniform(0.75, 1.333)
        #             ww = int(np.sqrt(targetArea * aspectR))
        #             hh = int(np.sqrt(targetArea / aspectR))
        #             if self.rng.uniform() < 0.5:
        #                 ww, hh = hh, ww
        #             if hh <= h and ww <= w:
        #                 x1 = 0 if w == ww else self.rng.randint(0, w - ww)
        #                 y1 = 0 if h == hh else self.rng.randint(0, h - hh)
        #                 out = img[y1:y1 + hh, x1:x1 + ww]
        #                 out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_CUBIC)
        #                 return out
        #         out = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        #         return out

        augmentors = [
            # Resize(),
            # imgaug.RandomOrderAug(
            #     [imgaug.Brightness(30, clip=False),
            #      imgaug.Contrast((0.8, 1.2), clip=False),
            #      imgaug.Saturation(0.4),
            #      # rgb-bgr conversion
            #      imgaug.Lighting(0.1,
            #                      eigval=[0.2175, 0.0188, 0.0045][::-1],
            #                      eigvec=np.array(
            #                          [[-0.5675, 0.7192, 0.4009],
            #                           [-0.5808, -0.0045, -0.8140],
            #                           [-0.5836, -0.6948, 0.4203]],
            #                          dtype='float32')[::-1, ::-1]
            #                      )]),
            # imgaug.Clip(),
            imgaug.Flip(horiz=True),
            imgaug.ToUint8()
        ]
    else:
        augmentors = [
            # imgaug.ResizeShortestEdge(256),
            # imgaug.CenterCrop((224, 224)),
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(8, multiprocessing.cpu_count()))
    # ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds


def get_config():
    dataset_train = get_data('train')
    dataset_val = get_data('val')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_train,
                            [BinaryClassificationStats('prediction', 'multilabel', prefix='train')]),
            InferenceRunner(dataset_val,
                [BinaryClassificationStats('prediction', 'multilabel', prefix='val')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(10, 1e-4), (20, 1e-5), (30, 1e-6)]),
            HumanHyperParamSetter('learning_rate')
        ],
        model=Model(),
        steps_per_epoch=STEPS_EPOCH,
        max_epoch=MAX_EPOCH,
    )


# def eval_on_ILSVRC12(model_file, data_dir):
#     ds = get_data('val')
#     pred_config = PredictConfig(
#         model=Model(),
#         session_init=get_model_loader(model_file),
#         input_names=['input', 'label'],
#         output_names=['wrong-top1', 'wrong-top5']
#     )
#     pred = SimpleDatasetPredictor(pred_config, ds)
#     acc1, acc5 = RatioCounter(), RatioCounter()
#     for o in pred.get_result():
#         batch_size = o[0].shape[0]
#         acc1.feed(o[0].sum(), batch_size)
#         acc5.feed(o[1].sum(), batch_size)
#     print("Top1 Error: {}".format(acc1.ratio))
#     print("Top5 Error: {}".format(acc5.ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    # parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=50, choices=[18, 34, 50, 101])
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    DEPTH = args.depth
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.eval:
        BATCH_SIZE = 128    # something that can run on one gpu
        # eval_on_ILSVRC12(args.load, args.data)
        sys.exit()

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU

    logger.auto_set_dir()
    config = get_config()
    if args.load:
        # config.session_init = SaverRestore(args.load)
        config.session_init = get_model_loader(args.load)
    config.nr_tower = NR_GPU
    SyncMultiGPUTrainer(config).train()
