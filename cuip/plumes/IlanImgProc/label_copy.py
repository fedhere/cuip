import scipy as sp
from scipy import ndimage
import pylab as pl
from skimage import filters


PLOT = False


def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def getregion(img):

    tmp2 = rebin(img, (img.shape[0]/4, img.shape[1]/4))

    imgtmp = tmp2.copy()
    imgtmp[tmp2<np.mean(tmp2) + 2.*np.var(tmp2)]=0
    imgtmp[imgtmp>0]=1
    

    labels, n_features = ndimage.measurements.label(imgtmp)
    labelsize = np.array([(labels==i).sum() for i in range(labels.max()+1)])
    #labtmp = {}
    #for i in range(labels.max()+1):
    #    labtmp[i] = labelsize[i]

    #zeros = np.zeros_like(imgtmp.flatten())
    labs = np.unique(labelsize)
    labdiff = np.diff(np.unique(labelsize))
    
    print len(labs),",",np.max(labdiff[:-1]),',',(labdiff>=50).sum(),',',n_features,',', np.mean(labdiff[:-1]),',',np.max(labdiff)
    return len(labs),np.max(labdiff[:-1]),(labdiff>=50).sum(),n_features, np.mean(labdiff[:-1])
    #for i,z in enumerate(labels.flat):
    #    if (labtmp[z]>20) * (labtmp[z]<1000) : 
            #print (labtmp[z])
    #       zeros[i]=labtmp[z]

    
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
       
    
