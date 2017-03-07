from __future__ import print_function, division
import numpy as np
import os
import sys
import glob
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import getCoords as gc

sys.path.append(os.path.realpath('..'))
import utils as utl
from configs import *


LOADSAVED = False
NPHOTOS = 40
DATE = '2013/10/26'
NPTS = 5
def get_dir_name(dir_path):
        '''return a list of string with names of directories 
        Arguments:
        dir_path: path to image directory
        '''

	#dirs = []
	#for item in os.listdir(dir_path):
#		if os.path.isdir(dir_path + item) == True: #ilan I had to change this, maybe you were in the local dir when running?
#			dirs.append(item)
	
	# writing it in a more pythonc way to improve efficiency and style
	dirs = np.array(os.listdir(dir_path))
	dirs = dirs[[os.path.isdir(dir_path + d) for d in dirs]]

	return dirs

def img2D(file_name):
	im = np.fromfile(file_name, np.uint8)
	im = im.astype(float)
	im *= 255 / im.max()
	im2d = (im.reshape([2160, 4096, 3]).sum(2) / 3.)
	im2d /= im2d.max()
	return im2d

# this function would cause troubles cause you are loading too many images ar once. this overloads the memory easily - lets use the util library I wrote instead which also finds the right size of the image so it does not need to be hardcoded in
def extract_images(path):
	img_list = []
	for filename in os.listdir(path):
		img_list.append(img2D(os.path.join(path,filename)))
	return img_list

def extract_times(time_string):
	'''extract time stamp from image name assuming 10 sec intervals
	Arguments:
	     time_string - string: the name of the file directory containing a time stamp
	'''
	base = dt.datetime.strptime(DATE + ' ' + time_string.split('/')[-1], 
				    '%Y/%m/%d %H.%M.%S') 
	time_vector = [base + dt.timedelta(seconds = i * DT)
		       for i in range(0, NPHOTOS)]
	# print(time_vector)
	return time_vector

def prepare_data(rootdir, time_dirs):

	image_list = []
	time_list = []
	for dr in [os.path.join(rootdir, fldr) for fldr in time_dirs]:
		image_list += ['/'.join([rootdir,fldr,d]) for d in os.listdir(dr)]
		time_list += extract_times(dr)

	time_list = np.array(time_list).flatten()
	image_list = np.array(image_list).flatten()

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
	

def getts(skyline, lim=-1):
	rootdir = os.getenv("ROOTDIR")
	if rootdir is None:
		print ("must set env variable ROOTDIR to plumes dir")
		sys.exit()
	#root = os.getenv("HOME") 'cuip/cuip/plumes/IlanImgProc/'  replacing this: it should be the local dir not a full path: my path to IlanImgProc is different! #lets set up an env variable CUIPLUMES
	rootdir = rootdir + "/" + DATE + "/"
	root = os.getenv("CUIPLUMES")
	if root is None:
		print ("must set env variable CUIPLUMES to plumes dir")
		sys.exit()
	#ilan I changed the data_path to out_path - call things data only if they contain the raw data
	out_path = root + '/output'
	if not os.path.isdir(out_path):
		os.makedirs(out_path)

	#TODO get better argument parser right now assume you can pass a skyline file
	
        if LOADSAVED :
		sl_idx, city_idx, sky_idx = load_index(root)
	else:
		sl_idx, city_idx, sky_idx = gc.img_points(skyline, 
							  n=NPTS,
							  show=False)
	time_dirs = get_dir_name(rootdir)
	time_dirs.sort()
	
	image_list, time_list =	prepare_data(rootdir, time_dirs)	
	pixels = np.hstack((sl_idx[::-1], 
			    city_idx[::-1], 
			    sky_idx[::-1]))

	rawimgs =  utl.RawImages(fl=image_list,  lim=lim, 
				 imsize = None, 
				 pixels = np.array(zip(pixels[1], 
						       pixels[0])))

	sky = pd.DataFrame(data = {"times":time_list[:lim]})
	skyline = pd.DataFrame(data = {"times":time_list[:lim]})
	city = pd.DataFrame(data = {"times":time_list[:lim]})

	dfs = {"skyline":skyline, "city":city, "sky":sky}
	for i in range(NPTS):
		for j,dfn in enumerate(dfs):
			df = dfs[dfn]
			df["R%03d"%i] = rawimgs.pixvals[:,i + j*NPTS,0] /\
			    np.median(rawimgs.pixvals[:,i + j*NPTS,0])
			df["G%03d"%i] = rawimgs.pixvals[:,i + j*NPTS,1] /\
			    np.median(rawimgs.pixvals[:,i + j*NPTS,1])
			df["B%03d"%i] = rawimgs.pixvals[:,i + j*NPTS,2] /\
			    np.median(rawimgs.pixvals[:,i + j*NPTS,2])
			df["x%03d"%i] = pixels[0,i]
			df["y%03d"%i] = pixels[1,i]
			
			df["mean%03d"%i] = (np.mean(
					np.array([df["R%03d"%i],
						  df["G%03d"%i],
						  df["B%03d"%i]]),
					axis=0))
			df["scene"] = [dfn]*len(df.times.values)
	return sky, skyline, city

if __name__ == '__main__':
	skyline = np.load(sys.argv[1])

	sky, skyline, city = getts(skyline)
	usefulcols = ["times"] + ["mean%03d"%i for i in range(NPTS)]
	
	sky[usefulcols].plot(x = "times", subplots = True, 
			     figsize = (9, 12), title = 'Sky')
	skyline[usefulcols].plot(x = "times", subplots = True, 
				 figsize = (9, 12), title = 'Skyline')
	city[usefulcols].plot(x = "times", subplots = True, 
			      figsize = (9, 12), title = 'City')

	sky[usefulcols].plot.hist(subplots = True, figsize = (9, 12), 
				  title = 'Sky', bins = 30)
	skyline[usefulcols].plot.hist(subplots = True, 
				      figsize = (9, 12), 
				      title = 'Skyline', bins = 30)
	city[usefulcols].plot.hist(subplots = True, figsize = (9, 12), 
				   title = 'City', bins = 30)
	plt.show()
	raw_input()
