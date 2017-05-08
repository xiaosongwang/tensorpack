#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: chestxray.py
# Author: Xiaosong Wang

import os
import glob
import cv2
import numpy as np
import cPickle as pickle

# from tensorpack import *
# from ...utils.fs import get_dataset_path
import tensorflow as tf
from ..base import RNGDataFlow

__all__ = ['CHESTXRAY14']
# DATA_URL = "https://console.cloud.google.com/storage/browser/gcs-public-data--nih/radiology_2017/Chest_X-Ray_CVPR17/"
# IMG_W, IMG_H = 481, 321
IMG_W, IMG_H = 512, 512

class CHESTXRAY14(RNGDataFlow):
    """
    `Berkeley Segmentation Data Set and Benchmarks 500 dataset
    <http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500>`_.

    Produce ``(image, label)`` pair, where ``image`` has shape (321, 481, 3(BGR)) and
    ranges in [0,255].
    ``Label`` is a floating point image of shape (321, 481) in range [0, 1].
    The value of each pixel is ``number of times it is annotated as edge / total number of annotators for this image``.
    """
    # _image_list = []
    def __init__(self, name, data_dir=None, shuffle=True):
        """
        Args:
            name (str): 'train', 'test', 'val'
            data_dir (str): a directory containing the original 'BSR' directory.
        """
        # check and download data
        if data_dir is None:
            self.data_dir = '/home/osboxes2/tensorpack_data/chestxray14_data'
        else:
            self.data_dir = data_dir
        self.data_root = os.path.join(self.data_dir, 'images')
        assert os.path.isdir(self.data_root)

        self.shuffle = shuffle
        assert name in ['train', 'test', 'val']
        self.name = name
        self._load(name)

        data_path = './'
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            self.word_to_idx = pickle.load(f)

        self.words = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'mass', 'nodule', 'pneumonia',
                      'pneumothorax', 'consolidation', 'edema', 'emphysema', 'fibrosis', 'pleural_thickening', 'hernia']

    def _load(self, name):
        image_list_file = os.path.join(self.data_dir,  name +'_label_20000n.txt')

        # read in lines from the image list
        self._image_list = tf.gfile.GFile(image_list_file).readlines()
        # for line in image_list:
        #     image_path = line.split(' ')[0]
        #     labels_str = line.split(' ')[1:-1]
        #     labels = [int(s) for s in labels_str]
        #     # print(image_path, labels)

    def size(self):
        return len(self._image_list)

    def _build_caption_vector(self, label, max_length=20):
        # n_examples = 1 # len(annotations)
        #captions = np.ndarray((max_length + 2)).astype(np.int32)

        word_to_idx = self.word_to_idx
        cap_vec = []
        count_NULL = 0
        cap_vec.append(word_to_idx['<START>'])
        for i, element in enumerate(label):
            #words = caption.split(" ")  # caption contrains only lower-case words
            if element:
                word = self.words[i]
                cap_vec.append(word_to_idx[word])
            else:
                count_NULL += 1
        if count_NULL == label.shape[0]:
            cap_vec.append(word_to_idx['no_finding'])
            count_NULL -= 1
        cap_vec.append(word_to_idx['<END>'])

        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>'])

        captions = np.asarray(cap_vec)
        # print "Finished building caption vectors"
        return captions

    def get_data(self):
        idxs = np.arange(len(self._image_list))
        # add_label_to_fname = (self.name != 'train' and self.dir_structure != 'original')
        # if self.shuffle:
        #     self.rng.shuffle(idxs)
        for k in idxs:
            line = self._image_list[k]
            fname = line.split(' ')[0]
            labels_str = line.split(' ')[1:-1]
            label = np.array([int(s) for s in labels_str])
            label = label.astype(np.int32)
            # label = tf.cast(label, tf.int32)
            # fname, label = self.imglist[k]
            # if add_label_to_fname:
            #     fname = os.path.join(self.full_dir, self.synset[label], fname)
            # else:
            fname = os.path.join(self.data_dir, fname)
            # print('#'+fname+'#')
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            # im = tf.image.per_image_standardization(im)
            # print(im.shape())
            # cv2.imshow("haha", im)
            # cv2.waitKey(1000)
            assert im is not None, fname  # --load ImageNet-ResNet50.npy
            if im.ndim == 2:
                im = np.expand_dims(im, 2).repeat(3, 2)
            # temp = [im, label]
            # print(temp[0].shape())
            # cv2.imshow("haha", temp[0])
            # im.set_shape([IMG_H, IMG_W, 3])
            # label.set_shape([14])
            caption = self._build_caption_vector(label, 14)
            yield [im, label, caption]


# try:
#     from scipy.io import loadmat
# except ImportError:
#     from ...utils.develop import create_dummy_class
#     BSDS500 = create_dummy_class('BSDS500', 'scipy.io')  # noqa

if __name__ == '__main__':
    a = CHESTXRAY14('val')
    for k in a.get_data():
        cv2.imshow("haha", k[0])
        cv2.waitKey(500)
