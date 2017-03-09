import numpy as np
import os
import sys
import argparse
import glob
import time
import pdb

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

# Load image
import os
if not os.path.isfile(sys.argv[4]):
	f_result = open(sys.argv[4], 'w')
	print >>f_result, 'in_img,style_img,out_img,in_img_mem,out_img_mem'
else:
	f_result = open(sys.argv[4],'a')
import sys
DIR_JPG_IN = sys.argv[1]
DIR_JPG_STYLE = sys.argv[2]
DIR_JPG_OUT = sys.argv[3]

for fn in os.listdir(DIR_JPG_IN):
    if os.path.isfile(os.path.join(DIR_JPG_IN,fn)):
        
        img = np.array(Image.open(os.path.join(DIR_JPG_IN,fn)))
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        mem_in = float(out['fc8-euclidean'][0])
	
	
        img = np.array(Image.open(os.path.join(DIR_JPG_OUT,fn)))
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        mem_out = float(out['fc8-euclidean'][0])
	
	result = "%s,%s,%s,%f,%f" % (os.path.join(DIR_JPG_IN,fn),DIR_JPG_STYLE,os.path.join(DIR_JPG_OUT,fn),mem_in, mem_out) 
        print result
	print >>f_result, result

