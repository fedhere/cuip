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

medfit_g = pd.DataFrame(index = range(nrows), columns = cols)  #empty dataframe to save fitted curves by image
var_g = np.zeros((30, 3))


#Gaussfit
for i in range(1,ncols):
    try:
        medfit_g.iloc[:,i-1], var_g[i-1]= gf.gaussfit(img_med.iloc[:,i], img_med.X, img_med.iloc[:,i].max(), \
            img_med.iloc[:,i].mean(), img_med.iloc[:,i].std())
    except RuntimeError: #because the fitting will not work for all images run through all to find those that don't work
        continue
    #var[i-1] = gf.gaussfit(img_med.iloc[:,i], img_med.X, img_med.iloc[:,i].max(), \
    #        img_med.iloc[:,i].mean(), img_med.iloc[:,i].std())[1]


#Lorentzfit

medfit_l = pd.DataFrame(index = range(nrows), columns = cols) 
var_l = np.zeros((30,2))

a = lambda x: 1 / (np.pi *max(x))
x0 = lambda x: sum(img_med.X * x) / sum(x)

for i in range(1,ncols):
    #try:
    medfit_l.iloc[:,i-1] = gf.lorentzfit(img_med.iloc[:,i], img_med.X, a(img_med.iloc[:,i]), \
            x0(img_med.iloc[:,i]))
    #except RuntimeError: 
    #    continue



###MINDIF.NPY fitting


#mdif_fit = pd.DataFrame(index = range(nrows), columns = cols) #empty dataframe to save fitted curves by image

#for i in range(1, ncols):
 #   try:
  #      mdif_fit.iloc[:,i-1] =  gf.gaussfit(img_mdif.iloc[:,i], img_mdif.X, img_mdif.iloc[:,i].max(), \
  #          img_mdif.iloc[:,i].mean(), img_mdif.iloc[:,i].std())
  #  except RuntimeError:
  #      continue


