import scipy as sp
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set()



def read_and_plot(data_csv, plot = False):
    """
    Take csv, clean it and plot all its features as subplots.
    Input is a data frame
    Return a Pandas Dataframe and the plot with plumes highlighted
    As default, no plot is generated, only the dataframe is returned
    """
    col_names = ['image', 'n_patches','2nd_largest_patch','patch_50+_pix','n_features','mean_patch_size', 'largest_patch']
    df = pd.read_csv(data_csv, header = None, names=col_names)
    df.loc[:,'image'] = np.array([int(l[12:16].strip().split()[0]) for l in df.loc[:,'image']])
    df.index = df.image
    df.drop('image', axis = 1, inplace = True)

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

    smooth = unsmooth.rolling(window = window, center = False, axis = 0).mean()
    smooth_sd = unsmooth.rolling(window = window, center = False, axis = 0).std()
    final = pd.DataFrame({'unsmooth':unsmooth, 'smooth':smooth, 'smooth_std':smooth_sd})
    
    if plot:
        ax = final.unsmooth.plot(figsize = (8,8), linestyle = 'none', marker = '.')
        final[['smooth', 'smooth_std']].plot(ax = ax)
        final.unsmooth[135:155].plot(ax = ax, marker = '.', linestyle = 'none', c = 'y')
        plt.title(' Rolling window 100 images', size = 15)
        plt.legend(['Images', 'Mean','St Dev', 'Plumes'], loc = 2)
        plt.ylabel(str(data_column.name + ' (Pixels)'), size = 13)
        plt.xlabel('Image', size = 13)
        plt.show()
        plt.savefig(str(data_column.name + '_roll.pdf'))
    
    if table:  
        return final
    
    
if __name__ == '__main__':
    
    df = read_and_plot(sys.argv[1], plot = False)
    column = df[sys.argv[2]]
    roll_and_plot(column, plot = True)
    
    
