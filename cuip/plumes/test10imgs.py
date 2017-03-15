import glob
import numpy as np
from pylab import *


allimg = np.zeros((10,1300,4096))
for i,f in enumerate(sorted(glob.glob("outputs/tmp_014?_median.npy"))):
    print (f)
    allimg[i] = np.load(f)[200:1500,:,:].mean(-1)


ion()
fig = figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax2.set_yscale('log')
draw()

for img in allimg:
   ax1.imshow(img, clim=(0,100), cmap="viridis")
   ax2.hist(img.flatten(), bins=range(0,80,2), alpha=0.6)
   raw_input()
   draw()
