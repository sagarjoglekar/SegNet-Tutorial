import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import argparse
import math
import cv2
import sys
import time
import pandas as pd
import pickle
from inpaintSegnetZones import inpaintMask
import multiprocessing as mp


# Make sure that caffe is on the python path:
caffe_root = '../../caffe-segnet/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu() 

modelDef = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/Example_Models/segnet_model_driving_webdemo.prototxt"
modelWeights = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/trained_weights/segnet_weights_driving_webdemo.caffemodel"
color_file = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/Scripts/camvid11.png"

Dir = "/datasets/sagarj/streetView/USAEasternAugImages_test/"

segmentFile = "/work/sagarj/Work/BellLabs/Data/segnetLabelsAugmented.pkl"
dictPickle = "/work/sagarj/Work/BellLabs/Data/similarityIds.pkl"

net = caffe.Net(modelDef,
                modelWeights,
                caffe.TEST)

#df = pd.read_csv(args.dataFrame)
with open(dictPickle, 'rb') as handle:
    df = pickle.load(handle)


input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape
label_colours = cv2.imread(color_file).astype('uint8')
input_shape = net.blobs['data'].data.shape

segmentedData = {}
count = 0

#pool = mp.Pool(processes=4)

labels = ['Sky', 'Building', 'Pole','Road_Marking','Road','Pavement','Tree','Sign_Symbol','Fence','Vehicle','Pedestrian', 'Bike']

for k in df:
    
    imageDirPath = Dir + k
    files = os.listdir(imageDirPath)
    segmentedData[k] = {}
    for fname in files:
        imagePath = imageDirPath + "/" + fname
        input_image_raw = caffe.io.load_image(imagePath)
        count+=1
        #name = args.outDir + "/" + k2 + ".png"
        start = time.time()
        input_image = caffe.io.resize_image(input_image_raw, (input_shape[2],input_shape[3]))
        input_image = input_image*255
        input_image = input_image.transpose((2,0,1))
        input_image = input_image[(2,1,0),:,:]
        input_image = np.asarray([input_image])
        input_image = np.repeat(input_image,input_shape[0],axis=0)
        net.blobs['data'].data[...] = input_image
        out = net.forward()
        end = time.time()
        print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'
        start = time.time()
        segmentation_ind = net.blobs['argmax'].data.copy()
        segLabels = np.squeeze(segmentation_ind)
        segmentedData[k][fname] = {}
        segmentedData[k][fname]['path'] = imagePath
        segmentedData[k][fname]['label'] = df[k]
        segmentedData[k][fname]['segmentedLabels'] = segLabels
        print segLabels.shape

        print 'Image %d at path %s' , count , imagePath

        end = time.time()
        print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'
            
            

        #cv2.imwrite(name, segmentation_rgb*255)

with open(segmentFile, 'wb') as handle:
    pickle.dump(segmentedData , handle, protocol=pickle.HIGHEST_PROTOCOL)