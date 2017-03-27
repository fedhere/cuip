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


#empty array to catch read images
img_med = np.zeros((30, 1300, 4096))
img_mdif = np.zeros((30, 1300, 4096))

for i,f in enumerate(sorted(glob.glob(os.path.join(PLUMES,'outputs/tmp_01[345][0-9]_median.npy')))):
    print(f)
    img_med[i] = np.load(f)[200:1500,:,:].mean(-1)

for i,f in enumerate(sorted(glob.glob(os.path.join(PLUMES, 'outputs/tmp_01[345][0-9]_mindif.npy')))):
    print(f)
    img_mdif[i] = np.load(f)[200:1500,:,:].mean(-1)

#set number of bins to 64 or Rice method
#save the X and Y (bin heights) components from histograms

BINS = 64#int(round(2*len(img_med[0].flatten())**(1./3.)))
y_med = np.zeros((30, BINS))
y_mdif = np.zeros((30, BINS))
x = np.zeros(BINS)
patches = np.zeros(BINS) #third element of histogram tuple NOT USED


for i in range(30):
    y_med[i], x, patches = plt.hist(img_med[i].flatten(), bins = BINS, log = True)
    y_mdif[i] = plt.hist(img_mdif[i].flatten(), bins = BINS, log = True)[0]
    

#Log10 of the bin heights
ymed_log = np.log10(y_med)
ymdif_log = np.log10(y_mdif)


#set -inf to zero
ymed_log[ymed_log == -np.inf] = 0.0
ymdif_log[ymdif_log == -np.inf] = 0.0

#create new X axis of the same dimension as number of bins. See doc on histograms
bins_cen = x[:-1] + 0.5*(x[1]-x[0])

#Create two new dataframes for simplicity. Already saved in IlanImgProc/

cols = ['img%d'%(i) for i in range(130,160)]

median_data = pd.DataFrame(ymed_log.T, columns = cols)
median_data.insert(0, 'X', bins_cen)
pd.DataFrame.to_csv(median_data, 'IlanImgProc/median_data.csv')


mindif_data = pd.DataFrame(ymdif_log.T, columns = cols)
mindif_data.insert(0, 'X', bins_cen)
pd.DataFrame.to_csv(mindif_data, 'IlanImgProc/mindif_data.csv')

