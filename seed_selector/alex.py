import numpy as np
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax
from skimage.transform import resize
from skimage import img_as_ubyte
import pickle
from functools import reduce


from lasagne.layers import InputLayer, Conv2DLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
import lasagne.nonlinearities


CROP_SIZE = (227, 227)

def build_model():
    net = {}
    net['data'] = InputLayer(shape=(None, 3, 227, 227))

    # conv1
    net['conv1'] = Conv2DLayer(
        net['data'],
        num_filters=96,
        filter_size=(11, 11),
        stride = 4,
        nonlinearity=lasagne.nonlinearities.rectify)

    
    # pool1
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)

    # norm1
    net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],
                                                     n=5,
                                                     alpha=0.0001/5.0,
                                                     beta = 0.75,
                                                     k=1)

    # conv2
    # The caffe reference model uses a parameter called group.
    # This parameter splits input to the convolutional layer.
    # The first half of the filters operate on the first half
    # of the input from the previous layer. Similarly, the
    # second half operate on the second half of the input.
    #
    # Lasagne does not have this group parameter, but we can
    # do it ourselves.
    #
    # see https://github.com/BVLC/caffe/issues/778
    # also see https://code.google.com/p/cuda-convnet/wiki/LayerParams
    
    # before conv2 split the data
    net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48,96), axis=1)

    # now do the convolutions
    net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad = 2)
    net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                     num_filters=128,
                                     filter_size=(5,5),
                                     pad = 2)

    # now combine
    net['conv2'] = concat((net['conv2_part1'],net['conv2_part2']),axis=1)
    
    # pool2
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride = 2)
    
    # norm2
    net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],
                                                     n=5,
                                                     alpha=0.0001/5.0,
                                                     beta = 0.75,
                                                     k=1)
    
    # conv3
    # no group
    net['conv3'] = Conv2DLayer(net['norm2'],
                               num_filters=384,
                               filter_size=(3, 3),
                               pad = 1)

    # conv4
    # group = 2
    net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192,384), axis=1)
    net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad = 1)
    net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],
                                     num_filters=192,
                                     filter_size=(3,3),
                                     pad = 1)
    net['conv4'] = concat((net['conv4_part1'],net['conv4_part2']),axis=1)
    
    # conv5
    # group 2
    net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192,384), axis=1)
    net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad = 1)
    net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],
                                     num_filters=128,
                                     filter_size=(3,3),
                                     pad = 1)
    net['conv5'] = concat((net['conv5_part1'],net['conv5_part2']),axis=1)
    
    # pool 5
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride = 2)

    # fc6
    net['fc6'] = DenseLayer(
            net['pool5'],num_units=4096,
            nonlinearity=lasagne.nonlinearities.rectify)

    # fc7
    net['fc7'] = DenseLayer(
        net['fc6'],
        num_units=4096,
        nonlinearity=lasagne.nonlinearities.rectify)

    # fc8
    net['fc8'] = DenseLayer(
        net['fc7'],
        num_units=1183,
        nonlinearity=lasagne.nonlinearities.softmax)
    
    
    params = np.load('alex_start_net.npy', encoding = 'latin1')
    lasagne.layers.set_all_param_values(net['fc8'], params)
 
    return net



MEAN_VALUES = np.array([104.0069879317889,  116.66876761696767,  122.6789143406786]) 

def normalize(img):
    img = np.moveaxis(img, -1, 0)
    new_img = img[::-1] - MEAN_VALUES.reshape((3, 1, 1))
    return new_img

def denormalize(img):
    old_img = np.moveaxis((img + MEAN_VALUES.reshape((3, 1, 1)))[::-1], 0, -1)
    return old_img.astype(np.uint8)


def crop(image, shift = 'random'):
    img_size = image.shape[:2]
    if shift == 'random':
        shift = np.array([np.random.randint(0, img_size[0] - CROP_SIZE[0], size = (1,)),
                          np.random.randint(0, img_size[1] - CROP_SIZE[1], size = (1,))])
    else :
        shift = (np.array(img_size) - np.array(CROP_SIZE)) / 2
    return image[shift[0]:(shift[0] + CROP_SIZE[0]), shift[1]:(shift[1] + CROP_SIZE[1])]

central_crop = lambda i: crop(i, shift = 'central')
resize_256 = lambda i: img_as_ubyte(resize(i, (256, 256)))

def apply_transforms_to_images(images, transforms):
    return np.array([reduce(lambda img, tr: tr(img), transforms, image) for image in images], dtype = np.float32)

def transform_train(images, transforms = [resize_256, crop, normalize]):
    return apply_transforms_to_images(images, transforms)

def transform_val(images, transforms = [resize_256, central_crop, normalize]):
    return apply_transforms_to_images(images, transforms)

def transform_test(images, transforms =  [resize_256, central_crop, normalize]):
    return apply_transforms_to_images(images, transforms)
