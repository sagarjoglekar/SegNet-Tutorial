import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time

# Make sure that caffe is on the python path:
caffe_root = '../../caffe-segnet/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
args = parser.parse_args()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape
label_colours = cv2.imread(args.colours).astype('uint8')
input_shape = net.blobs['data'].data.shape

input_image_raw = caffe.io.load_image(args.image)

input_image = caffe.io.resize_image(input_image_raw, (input_shape[2],input_shape[3]))
input_image = input_image*255
input_image = input_image.transpose((2,0,1))
input_image = input_image[(2,1,0),:,:]
input_image = np.asarray([input_image])
input_image = np.repeat(input_image,input_shape[0],axis=0)
net.blobs['data'].data[...] = input_image
start = time.time()
out = net.forward()
end = time.time()
print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

start = time.time()
#print net.blobs['prob'].data.shape
#squeeze = np.squeeze(net.blobs['prob'].data)
#segmentation_ind = np.argmax(squeeze , axis = 0)
#print segmentation_ind.shape
segmentation_ind = net.blobs['argmax'].data
segmentation_ind_3ch = np.resize(segmentation_ind,(3,input_shape[2],input_shape[3]))
segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

cv2.LUT(segmentation_ind_3ch,label_colours,segmentation_rgb)
segmentation_rgb = segmentation_rgb.astype(float)/255
print segmentation_ind
end = time.time()
print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

cv2.imwrite('out.png', segmentation_rgb*255) 