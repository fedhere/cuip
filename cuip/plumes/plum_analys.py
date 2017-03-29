import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
import gaussfit as gf
import seaborn as sns


img_med = pd.read_csv('IlanImgProc/median_data.csv', index_col = 0) #median dataframe
img_mdif = pd.read_csv('IlanImgProc/mindif_data.csv', index_col = 0) #mindif dataframe


nrows, ncols = img_med.shape
cols = ['img%d'%(i) for i in range(130,160)] #image ID vector


###MEDIAN.NPY fitting

mdfit = pd.DataFrame(index = range(nrows), columns = cols)  #empty dataframe to save fitted curves by image
mdcoeff= np.zeros((30, 3))


#Gaussfit
for i in range(1,ncols):
    try:
        mdfit.iloc[:,i-1], mdcoeff[i-1]= gf.gaussfit(img_med.iloc[:,i], img_med.X, img_med.iloc[:,i].max(), \
            img_med.iloc[:,i].mean(), img_med.iloc[:,i].std())
    except RuntimeError: #because the fitting will not work for all images run through all to find those that don't work
        mdcoeff[i-1] = [img_med.iloc[:,i].max(), 0, img_med.iloc[:,i].std()]


###MINDIF.NPY fitting


mnfit = pd.DataFrame(index = range(nrows), columns = cols) #empty dataframe to save fitted curves by image
mncoeff = np.zeros((30, 3))
for i in range(1, ncols):
    try:
        mnfit.iloc[:,i-1], mncoeff[i-1] =  gf.gaussfit(img_mdif.iloc[:,i], img_mdif.X, img_mdif.iloc[:,i].max(), \
                    img_mdif.iloc[:,i].mean(), img_mdif.iloc[:,i].std())
    except RuntimeError:
        mncoeff[i-1]  = [img_mdif.iloc[:,i].max(), 0, img_mdif.iloc[:,i].std()]



#real_img = img_med.drop('X', axis = 1) #filtering the X columnn (not an image)

#selecting the images with plumes and those without
mask =  np.ones(ncols-1, dtype=bool)
mask[13:18] = False

###Sigma values from original ditribution
#median
md_std = img_med.iloc[:,1:].std()

md_nplum = md_std[mask]
md_plum = md_std[~mask]

#mindif
mn_std = img_mdif.iloc[:,1:].std()

mn_nplum = mn_std[mask]
mn_plum = mn_std[~mask]


###Values from fitted distribution
#median

md_fitsd = mdcoeff[:,2]
mdfit_nplm = md_fitsd[mask]
mdfit_plm = md_fitsd[~mask]

#mindif
mn_fitsd = mncoeff[:,2]
mnfit_nplm = mn_fitsd[mask]
mnfit_plm = mn_fitsd[~mask]

###Plotting sigma for real distributions
#median
pl.ion()
fig, (ax1, ax2) = pl.subplots(2)
ax1.set_title('Real Image Variance (median)')
sns.set_style('whitegrid')
sns.distplot(md_nplum, hist = False, label = 'No Plume Sigmas', ax = ax1)
sns.distplot(md_plum, hist = False, label = 'Plume Sigmas', ax = ax1)

#media 
ax2.set_title('Real Image Variance (mindif)')
sns.distplot(mn_nplum, hist = False, label = 'No Plume Sigmas', ax = ax2)
sns.distplot(mn_plum, hist = False, label = 'Plume Sigmas', ax = ax2)
pl.show()


#mindif
#plt.figure()
#plt.title('Fitted Image Variance comparison (median)')
#sns.distplot(fit_nplm, hist = False, label = 'No Plume Sigmas')
#sns.distplot(fit_plm, hist = False, label = 'Plume Sigmas')


