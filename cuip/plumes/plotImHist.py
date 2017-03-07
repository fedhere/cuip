import numpy as np
import pylab as pl
import glob

#pl.ion()

fig = pl.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
flist = glob.glob("outputs/*npy")
fim = np.load(flist[0])
im = ax1.imshow(fim)
bars,bins = np.histogram(fim.flatten())
hist = ax2.bar(bins[:-1], bars, bins[1]-bins[0])

for f in flist:
    print (f)
    #fig = pl.figure()
    fim = np.load(f)
    im.set_data(fim)
    bars,bins = np.histogram(fim.flatten())
    for rect, h in zip(hist, bars):
            rect.set_height(h)
    #pl.draw()
    pl.savefig(f.replace("npy","pdf"))

    #raw_input()
    pl.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
