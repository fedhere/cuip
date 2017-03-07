import pandas as pd
import numpy as np
import seaborn as sns
import get_time_seriesFed as gt
import pylab as pl
import sys
import os

sys.path.append(os.path.realpath('..'))
import utils as utl
from configs import *

skyline =np.load("img1_skyline.npy")
sk, sl, ct = gt.getts(skyline, n=NPTS, lim=-1)
all3 = []
concatcols  = ["times","scene"]

for i in range(NPTS):
        all3 += [sk[concatcols + ["mean%03d"%i]]]
        all3 += [ct[concatcols + ["mean%03d"%i]]]

tmp = pd.concat(all3)
tmp["whichdp"] = np.where(~np.isnan(tmp[["mean%03d"%i for i in range(NPTS)]]))[1]
tmp["mean"] = np.nansum(tmp[["mean%03d"%i for i in range(NPTS)]], axis=1)

flatui = ["#9b59b6", "#3498db", "#95a5a6","#dcdcc6"]
sns.set_palette(flatui)
sns.violinplot(x="whichdp",y="mean", data=tmp, split=True, hue="scene")
pl.show()
raw_input()
