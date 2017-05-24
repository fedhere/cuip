# %load label_copy.py
import scipy as sp
from scipy import ndimage
import os
import glob
import numpy as np
import csv


PLOT = False


def rebin(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def getregion(img, threshold):
    """Extract labeled objects from images and create new features using the output.
    Input 2D array (image), resize it using rebin and threshold filter it
    using the default threshold: Mean + 3*Std
    
    The output are five numbers
    """
    tmp2 = rebin(img, (img.shape[0]/4, img.shape[1]/4))

    imgtmp = tmp2.copy()
    if threshold == 'm3sd':
        imgtmp[tmp2<np.mean(tmp2) + 3.*np.std(tmp2)]=0
    elif threshold == 'm4sd':
        imgtmp[tmp2<np.mean(tmp2) + 4.*np.std(tmp2)]=0
        
    else:
        raise ValueError, 'Please specify a valid threshold'
    
    imgtmp[imgtmp>0]=1
    
    labels, nlbl = ndimage.measurements.label(imgtmp)
    lbl_names = np.arange(1, nlbl+1)
    
    labelsize = ndimage.measurements.sum(imgtmp, labels, index = lbl_names)


    labs = np.unique(labelsize)
    labdiff = np.diff(np.unique(labelsize))
    
    centroid = ndimage.measurements.center_of_mass(imgtmp, labels, index = lbl_names)

    cent_x = np.zeros(len(centroid))
    cent_y = np.zeros(len(centroid))

    for i in range(len(centroid)):
        cent_x[i] = centroid[i][0]
        cent_y[i] = centroid[i][1]
   
    displ_x = np.diff(cent_x)
    displ_y = np.diff(cent_y)
    
    return(len(labs), np.max(labelsize), (labelsize>=50).sum(), nlbl,
           labelsize.mean(), labelsize.std(), displ_x.mean(), displ_y.mean(), displ_x.std(), displ_y.std())


def create_csv(ifile_path, ofile_path, threshold):
    tmp = np.sort(glob.glob(ifile_path))
    csvfile = open(ofile_path,'w')
    col_names = ['image', 'n_patches','largest_patch','patch_50+_pix',
                 'n_features','mean_patch_size', 'std_patch_size', 'mean_displ_X',
                 'mean_displ_Y', 'std_displ_X', 'std_displ_Y']
    
    writer = csv.writer(csvfile, delimiter=',',lineterminator='\n')
    writer.writerow(col_names)
    for t in tmp:
        try:
            img = np.load(t)[200:1300,:,:].mean(-1)
        except:
            pass
        line = [t[68:]]
        line.extend(getregion(img, threshold))  
        writer.writerow(line)
    #csvfile.close()
    
if __name__ == '__main__':
    plot = PLOT
    plumes = os.getenv('PLUMES_DATA')
    imgpath = os.path.join(plumes,'outputs/tmp_0*_median.npy')

    create_csv(imgpath,'output_csvs/mean+4sd_median_2.csv', threshold = 'm4sd')
