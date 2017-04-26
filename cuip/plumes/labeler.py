import scipy as sp
from scipy import ndimage
import glob
import pylab as pl
import numpy as np
PLOT = True

def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def getregion(img, plot=True):
    if plot:
        fig = pl.figure()
        ax1=fig.add_subplot(311); 
        ax2=fig.add_subplot(312); 
        ax3=fig.add_subplot(313); 

        ax1.imshow(img, interpolation='nearest', cmap='bone'); #pl.show()
        ax1.set_axis_off()



    tmp2 = rebin(img, (img.shape[0]/4, img.shape[1]/4))

    imgtmp = tmp2.copy()
    imgtmp[tmp2<np.mean(tmp2)+3.0*np.std(tmp2)]=0
    imgtmp[imgtmp>0]=1
    if plot:
        ax2.imshow(imgtmp, interpolation='nearest', cmap='viridis'); #pl.show()
        ax2.set_axis_off()

    labels = ndimage.measurements.label(imgtmp)[0]
    labelsize = np.array([(labels==i).sum() for i in range(labels.max()+1)])
    labtmp = {}
    for i in range(labels.max()+1):
        labtmp[i] = labelsize[i]

    zeros = np.zeros_like(imgtmp.flatten())
    labs = np.unique(labelsize)
    labdiff = np.diff(np.unique(labelsize))
    print len(labs), ",", np.max(labdiff[:-1]), ",", (labdiff>=50).sum()
    for i,z in enumerate(labels.flat):
        if (labtmp[z]>20) * (labtmp[z]<1000) : 
            #print (labtmp[z])
            zeros[i]=labtmp[z]
    if plot:
        ax3.imshow(zeros.reshape(tmp2.shape), interpolation='nearest', cmap='viridis'); 
        ax3.set_axis_off()
    #pl.show()
    return fig

if __name__ == '__main__':
    plot = PLOT
    tmp = np.sort(glob.glob("outputs/tmp_0*_median.npy"))
    for t in tmp:
        print t, ",",
        try:
            img = np.load(t)[200:1300,:,:].mean(-1)
        except:
            pass
            
        fig = getregion(img, plot)
        if plot:
            pl.savefig(t.replace('median.npy','median_labels.pdf')\
                           .replace('outputs','outputplots'))

        pl.close(fig)
