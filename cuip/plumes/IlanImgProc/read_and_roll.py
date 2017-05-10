import scipy as sp
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set()



def read_and_plot(file_path, plot = False):
    """
    Take csv, clean it and plot all its features as subplots.
    Input is a data frame
    Return a Pandas Dataframe and the plot with plumes highlighted
    As default, no plot is generated, only the dataframe is returned
    """
    #col_names = ['image', 'n_patches','2nd_largest_patch','patch_50+_pix','n_features','mean_patch_size', 'largest_patch']
    df = pd.read_csv(file_path, index_col = 'image')
    df.loc[:,'image'] = np.array([int(l[12:16].strip().split()[0]) for l in df.loc[:,'image']])
    df.index = df['image']
    del df['image']

    if plot:
        ax = df.plot(subplots = True,figsize = (8,8), title = str(data_csv))
        df.iloc[135:155,:].plot(subplots = True, ax = ax, color = 'k', legend = False)
        plt.savefig(str.join(data_csv,'.pdf'))
        plt.show()
    
    return df

def roll_and_plot(data_column, window = 100, plot = False, table = False):
    """
    Calculate and plot the rolling mean and rolling standard deviation for given dataframe.
    Input for column and datafile should be a string object.
    Return: Rolling mean and std table and plot saved as a pdf
    Default value for the window is 100.
    Plot is returned as default.
    Set table=True to get the rolling data frame
    """
    
    unsmooth = data_column

    x = np.arange(0,len(unsmooth),1)
    
    smooth_g = unsmooth.rolling(window = 40, center = True).mean()
    g = ndimage.filters.gaussian_filter(unsmooth, sigma = 20)
    smooth_sd = unsmooth.rolling(window = 40, center = True).std()
    f3sd = smooth_g + 3.*smooth_sd
    
    
    plt.figure(figsize = (7,5))
    plt.plot(x, unsmooth, linestyle = 'none', marker = '.')
    plt.plot(x, g, linestyle = 'none', marker='.')
    plt.plot(x, f3sd, linestyle = '--')
    plt.title('Second Largest Patch')
    plt.xlabel('Image')
    plt.ylabel('Patch Size (Pixels)')
    plt.legend(['Real', 'Gaussian', 'Gaussian + 3STD'])
    plt.show()
    
    
    
if __name__ == '__main__':
    
    df = read_and_plot('output_csvs/m3sd.csv', plot = False)
    x =
    column = df[]
    roll_and_plot(column, plot = True)
    
    
