import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PLUMES = os.getenv('PLUMES_DATA')
sys.path.append(PLUMES)
import gaussfit as gf


### Read in 30 images (130 to 159) from both median.npy and mindif.npy and store in arrays.
### Create histogram objects and extract X and Y values for each image
imgpath1 =  os.path.join(PLUMES,'outputlfs/tmp_*_median.npy')
imgpath2 =  os.path.join(PLUMES,'outputlfs/tmp_*_mindif.npy')

#print imgpath
imglist1 = sorted(glob.glob(imgpath1))
imglist2 = sorted(glob.glob(imgpath2))
        

#print imglist
nimgs = len(imglist1)

#empty array to catch read images
imgs = np.zeros((nimgs, 1300, 4096))
    
imgs_min = np.zeros((nimgs, 1300, 4096))


for i,f in enumerate(imglist1):
    print(f)
    imgs[i] = np.load(f)[200:1500,:,:].mean(-1)

for i,f in enumerate(imglist2):
    print(f)
    imgs_min[i] = np.load(f)[200:1500,:,:].mean(-1)

#set number of bins to 64 or Rice method
#save the X and Y (bin heights) components from histograms

BINS = 60#int(round(2*len(img_med[0].flatten())**(1./3.)))
y_md = np.zeros((nimgs, BINS))
y_mn = np.zeros((nimgs, BINS))
x = np.zeros(BINS)
patches = np.zeros(BINS) #third element of histogram tuple NOT USED


for i in range(nimgs):
    y_md[i], x, patches = plt.hist(imgs[i].flatten(), bins = BINS, log = True)
    y_mn[i] = plt.hist(imgs_min[i].flatten(), bins = BINS, log = True)[0]
    

#Log10 of the bin heights
ymd_log = np.log10(y_md)
ymn_log = np.log10(y_mn)


#set -inf to zero
ymd_log[np.isinf(ymd_log)] = 0.0
ymn_log[np.isinf(ymn_log)] = 0.0

#create new X axis of the same dimension as number of bins. See doc on histograms
bins_cen = x[:-1] + 0.5*(x[1]-x[0])

#Create two new dataframes for simplicity. Already saved in IlanImgProc/

#cols = ['img%d'%(i) for i in range(nimgs)]

#median_data = pd.DataFrame(ymd_log.T, columns = cols)
#median_data.insert(0, 'X', bins_cen)
#pd.DataFrame.to_csv(median_data, 'IlanImgProc/median_data.csv')


#mindif_data = pd.DataFrame(ymn_log.T, columns = cols)
#mindif_data.insert(0, 'X', bins_cen)
#spd.DataFrame.to_csv(mindif_data, 'IlanImgProc/mindif_data.csv')

