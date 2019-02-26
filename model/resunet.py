import numpy as np
import pandas as pd

import os

import random
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

import cv2
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input,  Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras import layers
from keras.losses import binary_crossentropy

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img


def convBn2d(input_tensor, filters, stage, block, kernel_size=(3, 3)):
    bn_axis = 3
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_name_base = 'conv2d' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base)(input_tensor)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base)(x)

    return x


def identity_block(input_tensor, filters, stage, block, kernel_size=(3, 3)):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = [filters] * 2

    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters1, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    x = layers.add([x, input_tensor])

    return x


def conv_block(input_tensor,
               filters,
               stage,
               block,
               strides=(2, 2),
               kernel_size=(3, 3)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = [filters] * 3
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters1, (2, 2), strides=strides,
                      padding='valid',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(input_tensor)
    shortcut = layers.Activation('relu')(shortcut)
    shortcut = layers.Conv2D(filters3, (2, 2), strides=strides,
                             padding='valid',
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])

    return x


def cse_block(prevlayer, prefix):
    mean = layers.Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    # mean = layers.Dropout(0.1)(mean)
    lin1 = layers.Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    # lin1 = layers.Dropout(0.1)(lin1)
    lin2 = layers.Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = layers.Multiply()([prevlayer, lin2])
    return x


def sse_block(prevlayer, prefix):
    #     conv = layers.Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
    #                   name=prefix + "_conv")(prevlayer)

    conv = layers.Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid',
                         strides=(1, 1),
                         name=prefix + "_conv")(prevlayer)
    conv = layers.Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv


def csse_block(x, prefix):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = layers.Add(name=prefix + "_csse_mul")([cse, sse])

    return x


def UResNet(input_shape, start_neurons,classes,hc=True,scse=False,block_num=5):
    '''
    input_layer is designed to (128,128,3)
    '''
    # 128 -> 128
    input_layer=Input(input_shape)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)

    conv1 = identity_block(conv1, filters=start_neurons * 1, stage=1, block='a')
    conv1 = identity_block(conv1, filters=start_neurons * 1, stage=1, block='b')
    conv1 = identity_block(conv1, filters=start_neurons * 1, stage=1, block='c')
    if(scse==True):
        conv1 = csse_block(conv1, 'stage1')

    # 128 -> 64
    conv2 = conv_block(conv1, filters=start_neurons * 2, stage=2, block='a')
    conv2 = identity_block(conv2, filters=start_neurons * 2, stage=2, block='b')
    conv2 = identity_block(conv2, filters=start_neurons * 2, stage=2, block='c')
    conv2 = identity_block(conv2, filters=start_neurons * 2, stage=2, block='d')
    if(scse==True):
        conv2 = csse_block(conv2, 'stage2')

    # 64 -> 32
    conv3 = conv_block(conv2, filters=start_neurons * 4, stage=3, block='a')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='b')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='c')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='d')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='e')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='f')
    if(scse==True):
        conv3 = csse_block(conv3, 'stage3')

    # 32 -> 16
    for i in range(block_num):
        conv4 = conv_block(conv3, filters=start_neurons * 8, stage=4, block=chr(ord('a')+i))
    if(scse==True):
        conv4 = csse_block(conv4, 'stage4')
    #conv4 = identity_block(conv4, filters=start_neurons * 8, stage=4, block='b')
    #conv4 = identity_block(conv4, filters=start_neurons * 8, stage=4, block='c')
    #conv4 = identity_block(conv4, filters=start_neurons * 8, stage=4, block='d')
    #conv4 = identity_block(conv4, filters=start_neurons * 8, stage=4, block='e')
    #conv4 = identity_block(conv4, filters=start_neurons * 8, stage=4, block='f')


    # 16 -> 8
    conv5 = conv_block(conv4, filters=start_neurons * 16, stage=-1, block='a')
    # dilation convolution
    # dilated_layer1 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same', dilation_rate=2)(conv5)
    # dilated_layer2 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same', dilation_rate=4)(dilated_layer1)
    # dilated_layer = layers.add([dilated_layer1, dilated_layer2])
    conv5 = convBn2d(conv5, start_neurons * 16, stage=-1, block='b')
    conv5 = convBn2d(conv5, start_neurons * 16, stage=-1, block='c')

    # 8 -> 16
    deconv5 = layers.UpSampling2D(size=(2, 2))(conv5)
    uconv5 = concatenate([deconv5, conv4])
    uconv5 = convBn2d(uconv5, start_neurons * 16, stage=0, block='a')
    uconv5 = convBn2d(uconv5, 64, stage=0, block='b')
    if(scse==True):
        uconv5 = csse_block(uconv5, 'stage0')

    # 16 -> 32
    deconv4 = layers.UpSampling2D(size=(2, 2))(uconv5)
    uconv4 = concatenate([deconv4, conv3])
    uconv4 = convBn2d(uconv4, start_neurons * 8, stage=5, block='a')
    uconv4 = convBn2d(uconv4, 64, stage=5, block='b')
    if(scse==True):
        uconv4 = csse_block(uconv4, 'stage5')

    # 32 -> 64
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = layers.UpSampling2D(size=(2, 2))(uconv4)
    uconv3 = concatenate([deconv3, conv2])
    uconv3 = convBn2d(uconv3, start_neurons * 4, stage=6, block='a')
    uconv3 = convBn2d(uconv3, 64, stage=6, block='b')
    if(scse==True):
        uconv3 = csse_block(uconv3, 'stage6')

    # 64 -> 128
    deconv2 = layers.UpSampling2D(size=(2, 2))(uconv3)
    uconv2 = concatenate([deconv2, conv1])
    uconv2 = convBn2d(uconv2, start_neurons * 2, stage=7, block='a')
    uconv2 = convBn2d(uconv2, 64, stage=7, block='b')
    if(scse==True):
        uconv2 = csse_block(uconv2, 'stage7')
    if(hc==True):
    # hypercolumn
        hypercolumn = concatenate([uconv2,
                               deconv2,
                               layers.UpSampling2D(size=(4, 4))(uconv4),
                               layers.UpSampling2D(size=(8, 8))(uconv5)])

        uconv2 = layers.Dropout(0.5)(hypercolumn)

    logits = Conv2D(64, (3, 3), padding='same', activation='relu')(uconv2)

    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(classes, (1, 1), padding="same", activation=None)(logits)
    output_layer = Activation('softmax')(output_layer_noActi)

    model=Model(input_layer,output_layer)
    model.summary()
    return model