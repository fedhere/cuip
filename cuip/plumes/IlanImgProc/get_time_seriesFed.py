import numpy as np
import os
import sys
import os.path
import glob
import pylab as pl
import datetime as dt
import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd


def get_dir_name(dir_path):
        '''return a list of string with names of directories 
        Arguments:
        dir_path: path to image directory
        '''
	dirs = []
	for item in os.listdir(dir_path):
		if os.path.isdir(item) == True:
			dirs.append(item)
	return dirs

def img2D(file_name):
	im = np.fromfile(file_name, np.uint8)
	im = im.astype(float)
	im *= 255 / im.max()
	im2d = (im.reshape([2160, 4096, 3]).sum(2) / 3.)
	im2d /= im2d.max()
	return im2d

def extract_images(path):
	img_list = []
	for filename in os.listdir(path):
		img_list.append(img2D(os.path.join(path,filename)))
	return img_list

def extract_times(time_string):
	n_photos = 40
	base = dt.datetime.strptime(time_string, '%H.%M.%S') 
	time_vector = [base + dt.timedelta(seconds = i*10) for i in range(0,n_photos)]
	return time_vector

def prepare_data(rootdir, time_dirs):

	image_list = []
	time_list = []
	for fldr in time_dirs:
		home = os.path.join(rootdir, fldr)
		image_list.append(extract_images(home))
		time_list.append(extract_times(home))

	time_list = [item for sublist in time_list for item in sublist]
	#image_list = [item for sublist in image_list for item in sublist]
	return image_list, time_list

def filter_data(image_list, index):

	temp_list = []
	for item in image_list: 
		for image in item:
			temp_list.append(image[index].tolist())
	
	data = pd.DataFrame(temp_list)
	data['times'] = times
    		
	return data

def load_index(root):
	sl_idx = np.load(root + 'skyline_idx.npy').tolist()
	city_idx  = np.load(root + 'city_idx.npy').tolist()
	sky_idx = np.load(root + 'sky_idx.npy').tolist()
	
	return sl_idx, city_idx, sky_idx
	

if __name__ == '__main__':
	rootdir = os.getenv("ROOTDIR")
	root = '/home/cusp/ir729/cuip/cuip/plumes/IlanImgProc/'
	
	data_path = '/home/cusp/ir729/data/'


#	sl_idx, city_idx, sky_idx = load_index(root)
	
#	time_dirs = get_dir_name(rootdir)
	#time_dirs.sort()

	#image_list, time_list =	prepare_data(rootdir, time_dirs)
	
	sky = pd.read_csv(data_path + 'data_sky.csv')
	skyline = pd.read_csv(data_path + 'data_skyline.csv')
	city = pd.read_csv(data_path + 'data_city.csv')

	sky.plot(x = sky.times, subplots = True, figsize = (9, 12), title = 'Sky')
	skyline.plot(x = skyline.times, subplots = True, figsize = (9, 12), title = 'Skyline')
	city.plot(x = city.times, subplots = True, figsize = (9, 12), title = 'City')

	sky.plot.hist(subplots = True, figsize = (9, 12), title = 'Sky', bins = 30)
	skyline.plot.hist(subplots = True, figsize = (9, 12), title = 'Skyline', bins = 30)
	city.plot.hist(subplots = True, figsize = (9, 12), title = 'City', bins = 30)
	plt.show()
	
	

	

	
	

