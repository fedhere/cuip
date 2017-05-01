import sys
import glob
import pandas as pd
import scipy as sp
from scipy import stats
import numpy as np

LIM = -1#10
moments = {2:'variance',
           3:'skew',
           4:'kurtosis'}

if sys.argv[1] == 'median':
    flist = sorted(glob.glob("outputs/tmp*median.npy"))[:LIM]
else:
    flist = sorted(glob.glob("outputs/tmp*mindif.npy"))[:LIM]    

nlist = len(flist)

immoments = pd.DataFrame()
immoments["name"] = ['']*nlist
immoments["imnumber"] = np.zeros(nlist)
immoments["mean"] = np.zeros(nlist)
immoments["median"] = np.zeros(nlist)
immoments["variance"] = np.zeros(nlist)
immoments["skew"] = np.zeros(nlist)
immoments["kurtosis"] = np.zeros(nlist)

for i,f in enumerate(flist):
    print(i,nlist)
    im = np.load(f).mean(-1)
    finim = im[250:1500,:][np.isfinite(np.log10(im[250:1500,:]))]
    immoments.set_value(i,"name", f)
    immoments.set_value(i,"imnumber", i)
    immoments.set_value(i,"mean", finim.mean())
    immoments.set_value(i,"median", np.median(finim))
    immoments.set_value(i,"variance", stats.moment(finim, 2))
    immoments.set_value(i,"skew", stats.moment(finim, 3))
    immoments.set_value(i,"kurtosis", stats.moment(finim, 4))

if sys.argv[1] == 'median':
    immoments.to_csv("moments_median.csv")
else:
    immoments.to_csv("moments_mindif.csv")
