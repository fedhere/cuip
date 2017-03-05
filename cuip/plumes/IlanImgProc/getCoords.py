import numpy as np
import scipy.ndimage as nd
from skyline import skyline
import pylab as pl

def img_points(skyline, n=5, save=False, seed=111, show=False, imgPath="img1.raw"):
	'''finds pixels on, ver and below the skyline
	Arguments:
	skyline - a np.ndarray 1D array containing the y pixel of the skyline for each x pixel in the image
	n - int (optional) number of pixels per region, default 5
	save - bool (optional) whether to save the n pixel coordinates to file (one file per section) - default False
	seed - int (optional) random sequence initialization seed default 100
	imgPath - optional: image path for plotting
	'''
	np.random.seed(seed)

	#selecting n x-coordinates)
	x_rand = np.random.choice(skyline.shape[0], n)

	sl = skyline[x_rand] # the skyline y pixels of the n x coordinates
	#city = [1000] * n ##what is this?
	#sky = [150] * n ## what is this?
	maxCity = int(2160 * 0.75)
	
	#chosing a random y for each x
	sky_idx  = np.array([[np.random.randint(0, sl[i] - 30) for i in range(n)], 
			    x_rand])
	city_idx = np.array([[np.random.randint(sl[i] + 30, maxCity) 
	                    for i in range(n)], x_rand])
	sl_idx = np.array([sl, x_rand])

	if save:
		np.save('skyline_idx', sl_idx)
		np.save('city_idx', city_idx)
		np.save('sky_idx', sky_idx)
	
	if show:
		pl.figure()
		pl.imshow(np.fromfile(imgPath, np.uint8).reshape([2160,
							   4096,3]))
		pl.plot(city_idx[1], city_idx[0], '.')				   
		pl.plot(sky_idx[1], sky_idx[0], '.')				   
		pl.plot(range(skyline.shape[0]), skyline, 'k.')
		pl.plot(sl_idx[1], sl_idx[0], '.')				   

		pl.show()

	return sl_idx, city_idx, sky_idx

if __name__ == '__main__':

	if len(sys.argv) > 1:
		imgskl = sys.argv[1] # assume the only argument is the skyline file
	skyline = np.load(imgskl)
	img_points(skyline, show=True)
    


