import numpy as np
import os
import sys
import argparse
import glob
import time
import pdb
from skimage.color import gray2rgb
from PIL import Image
import caffe

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img = cv2.equalizeHist(img)
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1]) not a RGB
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC) # make sure all the images are at the same size

    return img


caffe.set_device(0)
caffe.set_mode_gpu()

# Read model architecture and trained model's weight
net = caffe.Net('/home/gzen/memorability/memnet/deploy.prototxt', '/home/gzen/memorability/memnet/memnet.caffemodel', caffe.TEST)

# Load mean img
blob = caffe.proto.caffe_pb2.BlobProto()
data = open('/home/gzen/memorability/memnet/mean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
mean_array = arr[0].mean(1).mean(1)


# Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

import sys
DIR_JPG_IN = sys.argv[1]
FILE_OUT = sys.argv[2]

f = open(FILE_OUT, 'w')
print >>f, "img,mem_score"
for fn in os.listdir(DIR_JPG_IN):
    if os.path.isfile(os.path.join(DIR_JPG_IN,fn)):
        
        img = np.array(Image.open(os.path.join(DIR_JPG_IN,fn)))
	if len(img.shape) < 3:
		img = gray2rgb(img)
	if img.shape[2] == 4:
		img = img[:,:,:3]
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        mem_in = float(out['fc8-euclidean'][0])
	
	print >>f, "%s,%s" % (fn, mem_in)
f.close()
