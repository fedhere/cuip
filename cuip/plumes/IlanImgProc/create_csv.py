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
    
    labels, n_features = ndimage.measurements.label(imgtmp)
    
    labelsize = np.array([(labels==i).sum() for i in range(labels.max()+1)])
    lbl_names = np.unique(labels)
    labs = np.unique(labelsize)
    labdiff = np.diff(np.unique(labelsize))
   
    mx = np.zeros(len(lbl_names[1:]))
    my = np.zeros(len(lbl_names[1:]))    
    sdx = np.zeros(len(lbl_names[1:]))
    sdy = np.zeros(len(lbl_names[1:]))
    
    for i,j in enumerate(lbl_names[1:]):
        x,y = np.where(labels == j)
        #print i,len(x),len(y)
        
        mx[i] = x.mean()
        my[i] = y.mean()
        sdx[i] = x.std()
        sdy[i] = y.std()
    
    
    top_idx = sorted(range(len(labelsize[1:])), key=lambda i: labelsize[1:][i], reverse=True)[:2]
    
    mx1,mx2 = mx[top_idx[0]],mx[top_idx[1]]
    my1,my2 = my[top_idx[0]],my[top_idx[1]]
    sdx1,sdx2 = sdx[top_idx[0]], sdx[top_idx[1]]
    sdy1,sdy2 = sdy[top_idx[0]], sdy[top_idx[1]]
   

    return(len(labs),np.max(labdiff[:-1]),(labdiff>=50).sum(),n_features,
           np.mean(labdiff[:-1]),
           sdx1,sdy1,sdx2,sdy2,mx1,my1,mx2,my2)


def create_csv(ifile_path, ofile_path, threshold):
    tmp = np.sort(glob.glob(ifile_path))
    csvfile = open(ofile_path,'w')
    col_names = ['image', 'n_patches','largest_patch','patch_50+_pix',
                 'n_features','mean_patch_size','sdx_largest',
                 'sdy_largest','sdx_2nd_largest','sdy_2nd_largest',
                'center_X_largest','center_Y_largest','center_X_2largest','center_Y_2largest']
    
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
    imgpath = os.path.join(plumes,'outputs/tmp_0*_mindif.npy')

    create_csv(imgpath,'output_csvs/mean+3sd_mindif.csv', threshold = 'm3sd')
