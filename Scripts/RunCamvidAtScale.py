import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time
import pandas as pd
import pickle


# Make sure that caffe is on the python path:
caffe_root = '../../caffe-segnet/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_gpu() 

modelDef = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/Example_Models/segnet_model_driving_webdemo.prototxt"
modelWeights = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/trained_weights/segnet_weights_driving_webdemo.caffemodel"
color_file = "/work/sagarj/Work/BellLabs/Segnet-Tutorial/Scripts/camvid11.png"

segmentFile = "/segnetFringe2.pkl"
# Import arguments
parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, required=True)
#parser.add_argument('--weights', type=str, required=True)
#parser.add_argument('--colours', type=str, required=True)
#parser.add_argument('--image', type=str, required=False)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--outDir' , type=str , required=True)
args = parser.parse_args()

net = caffe.Net(modelDef,
                modelWeights,
                caffe.TEST)

#df = pd.read_csv(args.dataFrame)
with open(args.data, 'rb') as handle:
    df = pickle.load(handle)


input_shape = net.blobs['data'].data.shape
output_shape = net.blobs['argmax'].data.shape
label_colours = cv2.imread(color_file).astype('uint8')
input_shape = net.blobs['data'].data.shape

segmentedData = {}
count = 0

for k in df:
    ##Horribly bad way of managing data. Please think of something elegant
    imagesDict = {}
    imagesDict[k] = df[k]['origPath']
    t5Keys = df[k]['Top5Keys'][0]
    t5Paths = df[k]['Top5Paths'][0]
    #print t5Keys , t5Paths
    for k in range(len(t5Keys)):
        imagesDict[t5Keys[k]] = t5Paths[k]
    
    #print imagesDict
    for k in imagesDict:
        imagePath = imagesDict[k]
        input_image_raw = caffe.io.load_image(imagePath)
        count+=1
        name = args.outDir + "/" + k + ".png"
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
        segmentedData[k] = segmentation_ind
        print segmentation_ind
        
        print 'Image %d at path %s' , count , imagePath
        
        end = time.time()
        print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

        #cv2.imwrite(name, segmentation_rgb*255)

with open(args.outDir + segmentFile, 'wb') as handle:
    pickle.dump(segmentedData , handle, protocol=pickle.HIGHEST_PROTOCOL)
#segmentedDF.to_csv(args.outDir + "/segmentedDataframe.csv")
