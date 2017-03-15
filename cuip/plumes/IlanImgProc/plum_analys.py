import os
import numpy as np
import matplotlib.pyplot as plt

plumes = os.getenv('PLUMES_DATA') #plumes project, input files
npy_path = os.path.join(plumes, 'outputs') #dir with np arrays for anlysis
root = os.getenv('CUIP') # plumes repo


THRESHOLD = 60


def dist_expl(array_file, threshold = THRESHOLD, box = False, hist = False):
    '''
    
    Input: array file for distribution analysis after threshold filtering
    box: plots box plot of flat array, default = False
    hist: plots histogram of flat array, default = False
    
    '''

    f = np.load(array_file)
    f = f.flatten()
    out = f[f > THRESHOLD] #cut of values


    if box:
        plt.boxplot(out)
        plt.show()
    
    if hist:
        plt.hist(out)
        plt.show()


    

if __name__ == '__main__':
    
    f = 'tmp_0147_median.npy'
    file_path = os.path.join(npy_path, f)
    dist_expl(file_path, box = True)
    dist_expl(file_path, hist = True)
