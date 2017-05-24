import scipy as sp
from scipy import ndimage
import glob
import pylab as pl
import numpy as np
import sys

PLOT = True
SHOW = False

def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def weightedAvStd(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    indx = ~np.isnan(values)
    average = np.average(values[indx], weights=weights[indx])   
    variance = np.average((values[indx]-average)**2, weights=weights[indx])  # Fast and numerically precise
    return (average, np.sqrt(variance))

def getregion2(img, plot=True):

    tmp2 = rebin(img, (img.shape[0]/4, img.shape[1]/4))

    img= tmp2.copy()
    from sklearn.feature_extraction import image
    from sklearn.cluster import spectral_clustering
    graph = image.img_to_graph(img)

    graph.data = np.exp(-graph.data/graph.data.std())

    labels = spectral_clustering(graph)
    label_im = -np.ones(img.shape)
    label_im[mask] = labels
    print(label_im.shape)
    fig = pl.figure()
    ax1=fig.add_subplot(311); 
    ax2=fig.add_subplot(312); 
    ax3=fig.add_subplot(313); 

    ax1.imshow(img, interpolation='nearest', cmap='bone'); #pl.show()
    ax1.set_axis_off()
    

    ax2.imshow(label_im, interpolation='nearest', cmap='viridis'); #pl.show()
    ax2.set_axis_off()
    
    pl.show()
    
def getregion(img, plot=True):

    tmp2 = rebin(img, (img.shape[0]/4, img.shape[1]/4))

    imgtmp = tmp2.copy()

    imgtmp[tmp2<np.mean(tmp2)+3.0*np.std(tmp2)]=0
    imgtmp[imgtmp>0]=1


    labels, nfeat = ndimage.measurements.label(imgtmp)
    indx = range(0, nfeat + 1)
    
    labsize = np.array(ndimage.measurements.sum(imgtmp, labels=labels,
                                                index=indx))

    
    mask_size = labsize < 20
    remove_pixel = mask_size[labels]

    labels[mask_size[labels]] = 0
    indx = np.unique(labels[labels>0])    
    labsize = np.array(ndimage.measurements.sum(imgtmp, labels=labels,
                                                index=indx))    
    com = np.array(ndimage.measurements.center_of_mass(imgtmp, labels=labels,
                                                       index=indx))

    dists = com - np.array(imgtmp.shape[:2]) * 0.5
    dists = np.sqrt(dists[:,0]**2 + dists[:,1]**2)

    #pl.hist(dists)
    #pl.hist(dists/labsize)
    labs = np.unique(labsize)
    wa = weightedAvStd(dists, 1.0/labsize)
                                                
    print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
        nfeat, 
          labsize.mean(), labsize.std(), np.nanmean(dists),
          np.nanstd(dists), wa[0], wa[1]))

    labtmp = {}

    zeros = np.zeros_like(imgtmp.flatten())
    labdiff = np.diff(np.unique(labsize))
    labs = np.max(labdiff)

   
    if plot or labs > 50:
        fig = pl.figure()
        ax1=fig.add_subplot(311); 
        ax2=fig.add_subplot(312); 
        ax3=fig.add_subplot(313); 

        ax1.imshow(img, interpolation='nearest', cmap='bone'); #pl.show()
        ax1.set_axis_off()


        ax2.imshow(imgtmp, interpolation='nearest', cmap='viridis'); #pl.show()
        ax2.set_axis_off()
        ax3.imshow(labels, interpolation='nearest', cmap='viridis'); 
        #ax3.set_axis_off()
        if labs > 50 and SHOW:
            pl.show()
            
            #pl.show()
        return fig
    return 1
if 1: #__name__ == '__main__':
    plot = PLOT
    tmp = np.sort(glob.glob("outputlfs/tmp_0*_median.npy"))
    print("limage name,                  nfeat\tmeansz\tstdsz\tmndist\tstddist\twmdist\twmstddist")
    for t in tmp:
        print t,
        try:
            img = np.load(t)[200:1300,:,:].mean(-1)
        except:
            pass
            
        fig = getregion(img, plot)
        if plot:
            pl.savefig(t.replace('median.npy','median_labels.pdf')\
                           .replace('outputs','outputplots'))

        pl.close(fig)
    
