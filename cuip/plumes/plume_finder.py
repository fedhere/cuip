import os
import sys
import glob
import numpy as np
import gaussfit as gf
import pylab as pl
import pandas as pd

def goodOfFit(data, model):
    return sum((data - model)**2) / data.shape[0]

#MED = True


PLUMES = os.getenv('PLUMES_DATA')

if PLUMES is None:
    print ("must set PLUMES_DATA env variable to point to PLUME data")
    sys.exit()


### Read in 30 images (130 to 159) from both median.npy and mindif.npy and store in arrays.
### Create histogram objects and extract X and Y values for each image

def fitgauss():
    
    imgpath =     os.path.join(PLUMES,'outputlfs/tmp_*_median.npy')

    #print imgpath
    imglist = sorted(glob.glob(imgpath))
        

    #print imglist
    nimgs = len(imglist)
    #empty array to catch read images
    imgs = np.zeros((len(imglist), 1300, 4096))
    
    imgs_min = np.zeros((len(imglist), 1300, 4096))
    
    
    
    for i,f in enumerate(imglist):
        print(i,f)
        imgs[i] = np.load(f)[200:1500,:,:].mean(-1)
        f_min = f.replace("median", "mindif")
        if os.path.isfile(f_min):
            imgs_min[i] = np.load(f_min)[200:1500,:,:].mean(-1)                          
        #set number of bins to 64 or Rice method
        #save the X and Y (bin heights) components from histograms
        
    bins = np.arange(0, 51, 2.0)
    binscenter  = bins[:-1] + (bins[1]-bins[0]) * 0.5
    imgResids = np.zeros((nimgs, bins.shape[0]-1))
    
    x = np.zeros(bins.shape[0]-1)
    
    
    gofs = np.zeros((2, nimgs)) * np.nan
    meanstds = np.zeros((2, 2, nimgs)) * np.nan    
    
    for i in range(nimgs):
        for im in [imgs, imgs_min]:
            imgResids[i], x = np.histogram(imgs[i].flatten(), bins = bins)
            #Log10 of the bin heights
            y = np.log10(imgResids[i])
            y[np.isinf(y)] = 0.0
            #pl.bar(binscenter, y)
            
            try:
                distrib, coeffs = gf.gaussfit(y, binscenter,
                                              y.max(), 0, y.std())
            except        RuntimeError:
                coeffs   = [y.max(), 0, y.std()]
                
            gofs[0][i] = goodOfFit(y, distrib)
            meanstds[0][0][i] = coeffs[1]
            meanstds[0][1][i] = coeffs[2]            
            if (imgs_min[i] == 0).all():
                continue
            imgResids[i], x = np.histogram(imgs_min[i].flatten(), bins = bins)
            #Log10 of the bin heights
            y = np.log10(imgResids[i])
            y[np.isinf(y)] = 0.0
            #pl.bar(binscenter, y)
            
            try:
                distrib, coeffs = gf.gaussfit(y, binscenter, y.max(),
                                              0, y.std())
            except        RuntimeError:
                coeffs   = [y.max(), 0, y.std()]
                
            gofs[1][i] = goodOfFit(y, distrib)
            meanstds[1][0][i] = coeffs[1]
            meanstds[1][1][i] = coeffs[2]            
            #print coeffs, gofs[i]
            
            #pl.plot(binscenter, distrib)
            #print (coeffs)
            #pl.show()
    return imglist, gofs, meanstds

if __name__ == '__main__':
    imglist, gofs, coeffs = fitgauss()


    df = pd.DataFrame()
    df['imgmd'] = imglist
    df['imgmn'] = [im.replace("median", "mindif") for im in imglist]
    df['chisq_md'] = gofs[0]
    df['chisq_mn'] = gofs[1]
    df['mean_md'] = coeffs[0][0]
    df['mean_mn'] = coeffs[1][0]
    df['std_md'] = coeffs[0][1]
    df['std_mn'] = coeffs[1][1]        
    df.to_csv("plum_finder.csv")

