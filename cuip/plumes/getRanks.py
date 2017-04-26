import sys
import glob
import pandas as pd
import scipy as sp
from scipy import stats
import numpy as np

LIM = 10

if sys.argv[1] == 'median':
    flist = sorted(glob.glob("outputs/tmp*014*median.npy"))[:LIM]
else:
    flist = sorted(glob.glob("outputs/tmp*014*mindif.npy"))[:LIM]    

nlist = len(flist)


for i,f in enumerate(flist):
    print(i,nlist)
    im = np.load(f)[250:1500,:,:]
    rankedim = np.empty_like(im)
    for j in range(3):
        rankedim[:,:,j] = sp.stats.rankdata(im[:,:,j]).reshape(im[:,:,j].shape)
    


    np.save(f.replace("tmp","ranks"), rankedim)
