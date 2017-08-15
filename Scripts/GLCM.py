import numpy as np
import sys
from skimage.feature import greycomatrix, greycoprops
import cv2



def max_size(mat, ZERO=1):
    """Find the largest square of ZERO's in the matrix `mat`."""
    nrows, ncols = len(mat), (len(mat[0]) if mat else 0)
    if not (nrows and ncols): return 0 # empty matrix or rows
    counts = [[0]*ncols for _ in xrange(nrows)]
    for i in reversed(xrange(nrows)):     # for each row
        assert len(mat[i]) == ncols # matrix must be rectangular
        for j in reversed(xrange(ncols)): # for each element in the row
            if mat[i][j] != ZERO:
                counts[i][j] = (1 + min(
                    counts[i][j+1],  # east
                    counts[i+1][j],  # south
                    counts[i+1][j+1] # south-east
                    )) if i < (nrows - 1) and j < (ncols - 1) else 1 # edges
    return max(c for rows in counts for c in rows)


def computeGLCM (numpyImg ):
    g = greycomatrix(image, [5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=255,normed=True, symmetric=True)
    dissimilarity = greycoprops(g, 'correlation')
    return dissimilarity
    
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        print sys.argv[1]
        image = cv2.imread(sys.argv[1] , cv2.IMREAD_GRAYSCALE)
        
        print image.shape
        dis = computeGLCM(image)
        print dis.flatten()