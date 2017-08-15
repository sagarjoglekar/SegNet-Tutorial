import numpy as np
import cv2
import os
import sys
from skimage.feature import greycomatrix, greycoprops


def maximalSquare(matrix):
    m = matrix.shape[0]
    n = matrix.shape[1]
    npDp = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            if matrix[i][j]==0:
                npDp[i][j] = 0
            else:
                npDp[i][j] = 1
    #print npDp
    
    for i in xrange(1,m):
        for j in xrange(1,n):
            if matrix[i][j] == 1 : 
                npDp[i][j] = min(npDp[i-1][j],npDp[i][j-1],npDp[i-1][j-1])+1
            else: 
                npDp[i][j] = 0
    #print npDp
    ans = npDp.max()
    coord = np.where(npDp == ans)
    
    co_ords = []
    for x,y in zip(coord[0] , coord[1]):
        co_ords.append((x,y))
    print co_ords , ans
    return co_ords[-1] , int(ans-1)


def computeGLCM (srcPath , mask ):
    squareXY , squareSize = maximalSquare(mask)
    if squareSize < 20:
        return np.zeros((1,4))
    print "Square dimensions"
    print squareXY , squareSize
    if not os.path.exists(srcPath.strip()):
        print "Image not found"
        return
    name = srcPath.strip().split('/')[-1].split('.')[0]
    img = cv2.imread(srcPath.strip(),cv2.IMREAD_GRAYSCALE)
    img2 =  cv2.resize(img,  (mask.shape[1],mask.shape[0]))
    print img2.shape
    #Crop greyscale image using maximal overlapping square
    image = img2[int(squareXY[0]) - int(squareSize) : int(squareXY[0]) , int(squareXY[1]) - int(squareSize) : int(squareXY[1]) ]
    print image.shape
    g = greycomatrix(image, [3], [0, np.pi/4, np.pi/2, 3*np.pi/4],normed=True)
    dissimilarity = greycoprops(g, 'dissimilarity')
    
    return dissimilarity

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


if __name__ == "__main__":
    a = np.zeros((10,10))
    a[3:9,4:8] = 1
    #a[5:9 , 5:9] = 1
    
    print a
    
    maximalSquare(a)
