import numpy as np
import pylab as pl
import glob
import sys

#pl.ion()

if __name__=='__main__':
    if not len(sys.argv) == 2:
        print ('''usage: 
        python plotHist.py <filelist>
        where filelist is a path, possibly with whild-card syntax''')
    fig = pl.figure() # creating a single imag object for efficiency
    ax1 = fig.add_subplot(211) #top plot is the subtracted image
    ax2 = fig.add_subplot(212) #bottom plot is the brightness histogram

    flist = glob.glob(sys.argv[1]) 
    
    #create the first image out of the loop
    fim = np.load(flist[0])
    im = ax1.imshow(fim.mean(-1), cmap="viridis", clim=(0,255))
    bars,bins = np.histogram(fim.flatten(), bins=range(20,275,20))
    hist = ax2.bar(bins[:-1], bars, bins[1]-bins[0])
    nfiles = len(flist)
    for i,f in enumerate(flist):
        fout = f.replace("npy","pdf")
        print ("%d/%d:  %s"%(i, nfiles, fout))
        #upload data to figure in a loop: 

        #load data
        fim = np.load(f)

        #update diff image
        im.set_data(fim)

        #update histogram
        bars,bins = np.histogram(fim.flatten())
        for rect, h in zip(hist, bars):
            rect.set_height(h)

        pl.savefig(fout)

        #raw_input()
