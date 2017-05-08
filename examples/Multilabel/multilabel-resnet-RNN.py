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
import cPickle as pickle

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
CAPTION_SIZE = 16
STEPS_EPOCH = 5000 # 1000
MAX_EPOCH = 100


class Model(ModelDesc):
    def __init__(self):
        # rnn global parameters
        data_path = './'
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            self.word_to_idx = pickle.load(f)
        self.idx_to_word = {i: w for w, i in self.word_to_idx.iteritems()}
        self.prev2out = True
        self.ctx2out = True
        self.alpha_c = 1.0
        self.selector = True
        self.dropout = True
        self.V = len(self.word_to_idx)
        self.L = 16 * 16  # dim_feature[0]
        self.D = 1024  # dim_feature[1]
        self.M = 32  # dim_embed # not sure if it's used
        self.H = self.D + self.M  # dim_hidden
        self.T = 14 + 1  # n_time_step
        self._start = self.word_to_idx['<START>']
        self._null = self.word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        self.word_to_label = {'atelectasis':0, 'cardiomegaly':1, 'effusion':2, 'infiltration':3, 'mass':4, 'nodule':5,
                      'pneumonia':6, 'pneumothorax':7, 'consolidation':8, 'edema':9, 'emphysema':10, 'fibrosis':11,
                      'pleural_thickening':12, 'hernia':13}

        # # Place holder for features and captions
        # self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        # self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def _get_inputs(self):
        return [InputDesc(tf.uint8, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None, LABEL_SIZE], 'multilabel'),
                InputDesc(tf.int32, [None, CAPTION_SIZE], 'captions')]

    # rnn functions
    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)  # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])  # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M],
                                            initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _compute_wrong_label(self, sampled_word_list, label):
        # sampled_word_list_copy =
        pred_label = tf.zeros([BATCH_SIZE, label.shape[1]], tf.int32)
        _, list_T = sampled_word_list.shape
        list_N = BATCH_SIZE
        for i in range(list_N):
            for t in range(list_T):
                word = self.idx_to_word[sampled_word_list[i, t]]
                if word not in ['<START>', '<NULL>', '<END>']:
                    label_temp = self.word_to_label[word]
                    pred_label[i,label_temp] = 1
        # pred_label = tf.zeros([BATCH_SIZE, label.shape[1]], tf.int32, name='prediction')
        # _, list_T = sampled_word_list.shape
        # list_N = BATCH_SIZE
        # for i in range(list_N):
        #     for t in range(list_T):
        #         word = self.idx_to_word[sampled_word_list[i, t]]
        #         if word not in ['<START>', '<NULL>', '<END>']:
        #             label_temp = self.word_to_label[word]
        #             pred_label[i, label_temp] = 1
        #
        # wrong = tf.cast(tf.not_equal(pred_label, label), tf.float32)
        # wrong = tf.reduce_mean(wrong, name='train_error')
        return tf.cast(tf.not_equal(pred_label, label), tf.float32)


    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def _build_graph(self, inputs):
        image, label, captions = inputs
        image = tf.cast(image, tf.float32) * (1.0 / 255)

        # Wrong mean/std are used for compatibility with pre-trained models.
        # Should actually add a RGB-BGR conversion here.
        image_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        image_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        image = (image - image_mean) / image_std
        image = tf.transpose(image, [0, 3, 1, 2])

        ##### functions for resnet cnn model  #####
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
                      # transition layer added to resnet (w/o group = 2), output_size = 16*16*1024
                      .Conv2D('conv_CAM', 1024, 3, stride=1, nl=BNReLU)())
                      # .GlobalAvgPooling('gap')
                      # .FullyConnected('linearML', 14, nl=tf.identity)())

        features = tf.transpose(logits, perm=[0, 2, 3, 1])
        features = tf.reshape(features, [-1, 256, 1024])

        batch_size = tf.shape(features)[0]

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))
        x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        sampled_word_list = []
        if get_current_tower_context().is_training:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H)
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.H, forget_bias=1.0, input_size=None,
                state_is_tuple=True, activation=tf.tanh, reuse=True)


        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x[:, t, :], context], 1), state=[c, h])

            logits = self._decode_lstm(x[:, t, :], h, context, dropout=self.dropout, reuse=(t != 0))
            sampled_word = tf.argmax(logits, 1) # not sure
            sampled_word_list.append(sampled_word) # not sure
            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(None, captions_out[:, t], logits) * mask[:, t])



        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)  # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16. / 196 - alphas_all) ** 2)
            loss += alpha_reg

        xentropy = loss / tf.to_float(batch_size)

        costs = []
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        # xentropy = multilabel_class_balanced_sigmoid_cross_entropy(
        #     logits=logits, label=label, name='xentropy')
        costs.append(xentropy)

        sampled_word_list = tf.cast(tf.transpose(tf.stack(sampled_word_list), (1, 0)), tf.int32, name='prediction')
        wrong = tf.cast(tf.not_equal(sampled_word_list, captions[:, 1:16]), tf.float32)
        wrong = tf.reduce_mean(wrong, name='train_error')
        # wrong = self._compute_wrong_label(sampled_word_list, label)
        # pred_label = tf.zeros([BATCH_SIZE, label.shape[1]], tf.int32, name='prediction')
        # _, list_T = sampled_word_list.shape
        # list_N = BATCH_SIZE
        # for i in range(list_N):
        #     for t in range(list_T):
        #         word = self.idx_to_word[sampled_word_list[i, t]]
        #         if word not in ['<START>', '<NULL>', '<END>']:
        #             label_temp = self.word_to_label[word]
        #             pred_label[i, label_temp] = 1
        #
        # wrong = tf.cast(tf.not_equal(pred_label, label), tf.float32)
        # wrong = tf.reduce_mean(wrong, name='train_error')

        # pred = tf.cast(pred_label, tf.int32, name='prediction')
        # pred = tf.cast(tf.greater(logits, 0.5), tf.int32, name='prediction')
        # wrong = tf.cast(tf.not_equal(pred, label), tf.float32)
        # wrong, pred_label = self._compute_wrong(sampled_word_list, label)
        # pred_label = tf.cast(pred_label, tf.int32, name='prediction')

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

            # add_param_summary(('.*/W', ['histogram']))   # monitor W
            self.cost = tf.add_n(costs, name='cost')
            add_moving_summary(costs + [wrong, self.cost])
            # add_moving_summary(costs + [self.cost])

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.0001, summary=True)
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
    ds = AugmentImageComponent(ds, augmentors, copy=False)
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
                            [MultiLabelClassificationStats('prediction', 'captions', prefix='train')]),
            InferenceRunner(dataset_val,
                            [MultiLabelClassificationStats('prediction', 'captions', prefix='val')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(10, 1e-5), (20, 1e-6), (30, 1e-7)]),
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
