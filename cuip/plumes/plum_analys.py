import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gaussfit as gf

img_med = pd.read_csv('IlanImgProc/median_data.csv', index_col = 0) #median dataframe
img_mdif = pd.read_csv('IlanImgProc/mindif_data.csv', index_col = 0) #mindif dataframe


nrows, ncols = img_med.shape
cols = ['img%d'%(i) for i in range(130,160)] #image ID vector



###MEDIAN.NPY fitting

med_fit = pd.DataFrame(index = range(nrows), columns = cols)  #empty dataframe to save fitted curves by image

for i in range(1,ncols):
    try:
        med_fit.iloc[:,i-1] = gf.gaussfit(img_med.iloc[:,i], img_med.X, img_med.iloc[:,i].max(), 0, 10) \
            #img_med.iloc[:,i].mean(), img_med.iloc[:,i].std())
    except RuntimeError: #because the fitting will not work for all images run through all to find those that don't work
        continue

###MINDIF.NPY fitting


#mdif_fit = pd.DataFrame(index = range(nrows), columns = cols) #empty dataframe to save fitted curves by image

#for i in range(1, ncols):
 #   try:
  #      mdif_fit.iloc[:,i-1] =  gf.gaussfit(img_mdif.iloc[:,i], img_mdif.X, img_mdif.iloc[:,i].max(), \
  #          img_mdif.iloc[:,i].mean(), img_mdif.iloc[:,i].std())
  #  except RuntimeError:
  #      continue


