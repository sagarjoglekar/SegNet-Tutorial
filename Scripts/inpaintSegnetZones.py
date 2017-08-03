import numpy as np
import cv2
import os
import sys

def inpaintMask(srcPath , mask , destFolder , identifier):
    if not os.path.exists(srcPath.strip()):
        print "Image not found"
        return
    name = srcPath.strip().split('/')[-1].split('.')[0]
    img = cv2.imread(srcPath.strip())
    #print img.shape  
    #print mask.shape
    img2 =  cv2.resize(img,  (mask.shape[1],mask.shape[0])) 
    #print img2.shape
    dst = cv2.inpaint(img2,mask,3,cv2.INPAINT_NS)
    destPath = destFolder + "/" + name + '_' + str(identifier) + '.jpg' 
    cv2.imwrite(destPath,dst)
    print "Done inpainting %s for %s", srcPath, str(identifier) 
    return
