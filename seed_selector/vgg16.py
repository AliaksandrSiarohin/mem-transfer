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

CROP_SIZE = (224, 224)

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), name = 'input')
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False, name = 'conv1_1')
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False, name = 'conv1_2')
    net['pool1'] = PoolLayer(net['conv1_2'], 2, name = 'pool1')
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False, name = 'conv2_1')
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False, name = 'conv2_2')
    net['pool2'] = PoolLayer(net['conv2_2'], 2, name = 'pool2')
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False, name = 'conv3_1')
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False, name = 'conv3_2')
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False, name = 'conv3_3')
    net['pool3'] = PoolLayer(net['conv3_3'], 2, name = 'pool3')
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False, name = 'conv4_1')
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False, name = 'conv4_2')
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False, name = 'conv4_3')
    net['pool4'] = PoolLayer(net['conv4_3'], 2, name = 'pool4')
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False, name = 'conv5_1')
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False, name = 'conv5_2')
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False, name = 'conv5_3')
    net['pool5'] = PoolLayer(net['conv5_3'], 2, name = 'pool5')
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096, name = 'fc6')
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5, name = 'fc6_dropout')
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096, name = 'fc7')
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5, name = 'fc7_dropout')
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None, name = 'fc8')    
    
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    
    weightFile = open('vgg16.pkl', 'rb')
    values = pickle.load(weightFile, encoding='latin1')['param values']
    lasagne.layers.set_all_param_values(net['prob'], values)    

    return net

MEAN_VALUES = np.array([103.939, 116.779, 123.68]) 

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
