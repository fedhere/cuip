import numpy as np
import os
import scipy as sp
from scipy import ndimage
import glob
import csv
import sys


PLOT = False


def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def getregion(img, threshold = 'm3sd'):
    """Extract labeled objects from images and create new features using the output.
    Input 2D array (image), resize it using rebin and threshold filter it
    using the default threshold: Mean + 3*Std
    Other thresholds will be added
    The output are five numbers
    """
    tmp2 = rebin(img, (img.shape[0]/4, img.shape[1]/4))

    imgtmp = tmp2.copy()
    if threshold == 'm3sd':
        imgtmp[tmp2<np.mean(tmp2) + 3.*np.std(tmp2)]=0
        
    else:
        raise ValueError, 'Please specify a valid threshold'
    
    imgtmp[imgtmp>0]=1
    labels, n_features = ndimage.measurements.label(imgtmp)
    labelsize = np.array([(labels==i).sum() for i in range(labels.max()+1)])

    labs = np.unique(labelsize)
    labdiff = np.diff(np.unique(labelsize))
    
    #l= [len(labs),np.max(labdiff[:-1]),(labdiff>=50).sum(),n_features,np.mean(labdiff[:-1]),np.max(labdiff)]
    #print l
    return len(labs),np.max(labdiff[:-1]),(labdiff>=50).sum(),n_features,np.mean(labdiff[:-1]),np.max(labdiff)


def create_csv(ifile_path, ofile_path):
    tmp = np.sort(glob.glob(ifile_path))
    csvfile = open(ofile_path,'w')
    col_names = ['image', 'n_patches','2nd_largest_patch','patch_50+_pix',
                 'n_features','mean_patch_size', 'largest_patch']
    
    writer = csv.writer(csvfile, delimiter=',',lineterminator='\n')
    writer.writerow(col_names)
    for t in tmp:
        try:
            img = np.load(t)[200:1300,:,:].mean(-1)
        except:
            pass
        line = [t[68:]]
        line.extend(getregion(img, threshold = 'm3sd'))  
        writer.writerow(line)
    csvfile.close()
    

    
if __name__ == '__main__':
    plot = PLOT
    plumes = os.getenv('PLUMES_DATA')
    imgpath = os.path.join(plumes,'outputs/tmp_0*_median.npy')
    create_csv(imgpath,'output_csvs/test.py')
    
            

