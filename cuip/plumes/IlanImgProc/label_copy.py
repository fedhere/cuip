import scipy as sp
from scipy import ndimage
import glob
import pylab as pl
import numpy as np
import os

PLOT = False


def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def getregion(img):
    
    tmp2 = rebin(img, (img.shape[0]/4, img.shape[1]/4))

    imgtmp = tmp2.copy()
    imgtmp[tmp2<6]=0
    imgtmp[imgtmp>0]=1
    
    labels, nlabs = ndimage.measurements.label(imgtmp)
    labelsize = np.array([(labels==i).sum() for i in range(labels.max()+1)])
    #labtmp = {}
    #for i in range(labels.max()+1):
    #    labtmp[i] = labelsize[i]

    #zeros = np.zeros_like(imgtmp.flatten())
    labs = np.unique(labelsize)
    labdiff = np.diff(np.unique(labelsize))
    
    print len(labs),",", np.max(labdiff[:-1]),",", (labdiff>=50).sum(),',',nlabs,',',np.mean(labdiff[:-1]),',',np.max(labdiff)
    #for i,z in enumerate(labels.flat):
        #if (labtmp[z]>20) * (labtmp[z]<1000) : 
            #print (labtmp[z])
            #zeros[i]=labtmp[z]
            
            
def read_and_plot(data_csv, plot = True):
    
    col_names = ['image', 'n_patches','sec_largest_patch','patch_50+_pix','n_features','mean_patch_size', 'largest_patch']
    df = pd.read_csv(data_csv, header = None, names=col_names)
    df.loc[:,'image'] = np.array([int(l[12:16].strip().split()[0]) for l in df.loc[:,'image']])
    df.index = df.image
    df.drop('image', axis = 1, inplace = True)
    if plot:
        fig = plt.figure(figsize = (20,20))
        ax = df.iloc[0:300,:].plot(subplots = True)
        df.iloc[135:155,:].plot(subplots = True, ax = ax, color = 'k', legend = False)
        plt.title(str(data_csv))
        plt.show()
    
    return df
    
if __name__ == '__main__':
    plot = PLOT
    plumes = os.getenv('PLUMES_DATA')
    imgpath = os.path.join(plumes,'outputs/tmp_0*_median.npy')
    tmp = np.sort(glob.glob(imgpath))

    for t in tmp:
        print t[68:],",",
        try:
            img = np.load(t)[200:1300,:,:].mean(-1)
        except:
            pass
            
        getregion(img)
        #if plot:
         #   pl.savefig(t.replace('median.npy','median_labels.pdf')\
                           #.replace('outputs','outputplots'))

        #pl.close(fig)
    
