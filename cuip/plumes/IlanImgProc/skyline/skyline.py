#Import packages and libraries

import os
import sys
import numpy as np
import pylab as pl
import scipy.ndimage as nd
from skimage import feature
from skimage import filters as skfl
import matplotlib.pyplot as plt



#Define showme() function 

def showme(image, ax=None, cmap=None):
    if ax is None:
        ax = pl.figure(figsize=(11,11)).add_subplot(111)

    if cmap is None:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap=cmap)
    ax.axis('off')

def skyline(image, imgname, plotme=False):
    '''Take image, increase contrast and smooth. 
    Apply Sobel filter and find the skyline using gradient by column.
    Return max gradient closest to the top of the image.
    '''
    
    img = image.astype(float) 

    img *= 255 / img.max() 
	
    img2d = (img.reshape([2160, 4096, 3]).sum(2) / 3.)
    img2d /= img2d.max()

    smoothImg = nd.filters.gaussian_filter(img2d, [8, 8]) #apply gaussian filter to smooth the image and improve the edge detection

    smoothImgEn = smoothImg[:,:] * (np.atleast_2d(np.linspace(1, 0, smoothImg.shape[0])).T)**2 #increasing the contrast of the top section of the image

    edge = skfl.sobel(smoothImgEn) #new image with sobel edges

    edge[edge < 0.0025] = 0.0 #filtering after reviewing histogra of pixels
    #showme(edge, cmap = 'gray')

    rows = edge.shape[0]
    cols = edge.shape[1]
	
    grad_max = []
    for i in range(cols):
       	grad = np.gradient(edge[:,i]) # take the gradient of each column
        top_min = np.argpartition(grad, -20)[-20:].min() # select the minimum index (highest point) of 20 maximum gradient candidates
        if top_min > 732: #vertical threshold set by inspection
            top_min = 732
        grad_max.append(top_min)

    if plotme:
        showme(edge, cmap = 'gray')
        plt.plot(np.arange(cols), grad_max, 'g-', ms = 1.5)
        plt.axhline(y = 732)
        plt.show()
 
    np.save(imgname.split('.')[0] + '_skyline', grad_max)
    return grad_max


#path = 'cuip/cuip/plumes/IlanImgProc/'

if __name__ == '__main__':
    imgname = 'img1.raw'
    rawimg = np.fromfile(imgname, np.uint8)
    skl = np.load('img1_skyline.npy')
    #skyline(rawimg, imgname, plotme = True)
    print (skl)



