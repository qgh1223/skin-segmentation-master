from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K


CIFAR_TH_WEIGHTS_PATH = ''
CIFAR_TF_WEIGHTS_PATH = ''
CIFAR_TH_WEIGHTS_PATH_NO_TOP = ''
CIFAR_TF_WEIGHTS_PATH_NO_TOP = ''

IMAGENET_TH_WEIGHTS_PATH = ''
IMAGENET_TF_WEIGHTS_PATH = ''
IMAGENET_TH_WEIGHTS_PATH_NO_TOP = ''
IMAGENET_TF_WEIGHTS_PATH_NO_TOP = ''


def __initial_conv_block(input,weight_decay = 5e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(64, (3, 3), padding = 'same', use_bias = False, kernel_initializer = 'he_normal',
               kernel_regularizer = l2(weight_decay))(input)

    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation('relu')(x)

    return x

def conv_block(x,size):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(size, (3, 3), padding = 'same', use_bias = False)(x)

    return x

def __initial_conv_block_imagenet(input, weight_decay = 5e-4):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(64, (7, 7), padding = 'same', use_bias = False, kernel_initializer = 'he_normal',
               kernel_regularizer = l2(weight_decay), strides = (2, 2))(input)

    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)

    return x

def __grouped_convolution_block(input, grouped_channels, cardinality, strides,
                                weight_decay = 5e-4):

    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    group_list = []

    if cardinality == 1:
        x = Conv2D(grouped_channels, (3, 3), padding = 'same', use_bias = False, strides = (strides, strides),
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay))(init)
        x = BatchNormalization(axis = channel_axis)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

        x = Conv2D(grouped_channels, (3, 3), padding = 'same', use_bias = False, strides = (strides, strides),
                   kernel_initializer = 'he_normal', kernel_regularizer = l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis = channel_axis)
    x = BatchNormalization(axis = channel_axis)(group_merge)
    x = Activation('relu')(x)

    return x


def __bottleneck_block(input, filters = 64, cardinality = 8, strides = 1, weight_decay = 5e-4):
    init = input
    grouped_channels = int(filters / cardinality)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding = 'same', strides = (strides, strides),
                          use_bias = False, kernel_initializer='he_normal', kernel_regularizer = l2(weight_decay))(init)
            init = BatchNormalization(axis = channel_axis)(init)

    else:
        if init._keras_shape[-1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis = channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding = 'same', use_bias = False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)

    x = BatchNormalization(axis = channel_axis)(x)
    x = Activation('relu')(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding = 'same', use_bias = False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

    x = BatchNormalization(axis = channel_axis)(x)

    x = add([init, x])

    x = Activation('relu')(x)

    return x

def __create_res_next(nb_classes, img_input, include_top, depth = 29, cardinality = 8,
                      width = 4, weight_decay = 5e-4, pooling = None):

    if type(depth) is list or type(depth) is tuple:
        #if a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        #otherwise, default to 3 blocks
        N = [(depth - 2) // 9 for _ in range(3)]

    filters = cardinality * width
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2

    x = __initial_conv_block(img_input, weight_decay)

    #block 1 (no pooling)
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides = 1, weight_decay = weight_decay)

    N = N[1:] #remove the first block from block definition list
    filters_list = filters_list[1:] #remove the first filter from filter list

    #block 2 (input_size becomes half)
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 2,
                                       weight_decay = weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 1,
                                       weight_decay = weight_decay)


    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, use_bias = False, kernel_initializer='he_normal',
                  kernel_regularizer = l2(weight_decay), activation = 'softmax')(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x


def UResXNet(input_shape,classes,cardinality = 32,
                               depth=[3, 4, 6, 3],
                               width = 4, weight_decay = 5e-4,  pooling = None):

    if type(depth) is list or type(depth) is tuple:
        #if a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        #otherwise, default to 3 blocks
        N = [(depth - 2) // 9 for _ in range(3)]

    img_input=Input(input_shape)
    filters = cardinality * width
    filters_list = []
    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2

    x = __initial_conv_block(img_input, weight_decay)
    convlist=[]

    #block 1 (no pooling)
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides = 1, weight_decay = weight_decay)

    convlist.append(x)

    N = N[1:] #remove the first block from block definition list
    filters_list = filters_list[1:] #remove the first filter from filter list

    #block 2 (input_size becomes half)

    for block_idx, n_i in enumerate(N[1:]):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 2,
                                       weight_decay = weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 1,
                                       weight_decay = weight_decay)

        convlist.append(x)

    x=Conv2D(filters_list[-1],(3,3),strides=(2,2),padding='same')(x)

    for i,n_i in enumerate(N):
        x=UpSampling2D((2,2))(x)
        x=concatenate([x,convlist[len(N)-i-1]])
        if(i!=4):
            x=conv_block(x,filters_list[len(N)-i-1])
        else:
            x=conv_block(x,128)
        x=conv_block(x,64)

    x=Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x=Conv2D(classes,(3,3),padding='same',activation='softmax')(x)

    model=Model(img_input,x)
    model.summary()
    return x

#__create_res_next_imagenet((256,256,3),2)
