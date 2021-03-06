"""
detect_match.py
Author: Chris Prince [cmp670@nyu.edu]
Date: 25 May 2016
"""

import os
from math import asin
import numpy as np
import pylab as pl
import cv2
from scipy.ndimage.filters import convolve

# Globals
CAMHEIGHT, CAMWIDTH = (2160, 4096)


def loadRAW(f):
    return np.fromfile(f, dtype=np.uint8).reshape(CAMHEIGHT,CAMWIDTH,3)[:,:,::-1]


def drawMatches(img1, kp1, img2, kp2, matches):
    """
        Adapted from Ray Phan's code at
        http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python

        Implementation of cv2.drawMatches as OpenCV 2.4.9
        does not have this function available but it's supported in
        OpenCV 3.0.0

        This function takes in two images with their associated
        keypoints, as well as a list of DMatch data structure (matches)
        that contains which keypoints matched in which images.

        An image will be produced where a montage is shown with
        the first image followed by the second image beside it.

        Keypoints are delineated with circles, while lines are connected
        between matching keypoints.

        img1,img2 - Grayscale
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
                  detection algorithms
        matches - A list of matches of corresponding keypoints through any
                  OpenCV keypoint matching algorithm
    """
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')
    # Place the first image to the left
    out[:rows1, :cols1, :] = img1 #np.dstack([img1, img1, img1])
    # Place the next image to the right of it
    out[:rows2, cols1:cols1+cols2, :] = img2 #np.dstack([img2, img2, img2])
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (255, 0, 0), 1)
        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        clr = tuple(np.random.randint(0, 255, 3))
        cv2.line(out, (int(x1), int(y1)),
                 (int(x2)+cols1, int(y2)), clr, 1)
    # Show the image
    return out


def feature_match(img1, img2, detectAlgo=cv2.SIFT(), matchAlgo='bf',
                  bfNorm=cv2.NORM_L2,
                  flannIndexParams=dict(algorithm=0,  # =cv2.FLANN_INDEX_KDTREE
                                        trees=5),
                  flannSearchParams=dict(checks=20)
                  ):
    """
    Find features in images img1 and img2 using a detection algorithm (default
    is cv2.SIFT), then match them according to a brute force ('bf') or fast
    approximation ('flann') method. Returns the features and the matches.
    """

    support_detect = ['sift', 'surf', 'orb']
    support_match = ['bf', 'flann']

    # The actual feature detection computations
    kp1, des1 = detectAlgo.detectAndCompute(img1, None)
    kp2, des2 = detectAlgo.detectAndCompute(img2, None)

    if matchAlgo not in support_match:
        errmsg = 'Matching algorithm {} not supported ' \
                 '(must be one of {})'.format(
                  matchAlgo, ', '.join(support_match))
        raise ValueError(errmsg)
    elif matchAlgo == 'bf':
        # bf = cv2.BFMatcher(cv2.NORM_L2) #cv2.NORM_HAMMING)
        # TODO: decide whether to allow norm to be chosen by user or make
        # dependent on algorithm
        matcher = cv2.BFMatcher(bfNorm)
    elif matchAlgo == 'flann':
        # FLANN is an optimized matcher (versus brute force), so should play
        # with this as well. Maybe brute force is better if we go with multiple
        # subsamples to register image (instead of entire image)
        matcher = cv2.FlannBasedMatcher(flannIndexParams, flannSearchParams)
    else:
        print 'should not have gotten here!'
        raise Exception('wow')

    # The actual matching computation
    matches = matcher.knnMatch(des1, des2, k=2)

    # Return the keypoints and matches as a tuple
    return (kp1, des1, kp2, des2, matches)


def match_des(des1, des2, matchAlgo='bf',
                  bfNorm=cv2.NORM_L2,
                  flannIndexParams=dict(algorithm=0,  # =cv2.FLANN_INDEX_KDTREE
                                        trees=5),
                  flannSearchParams=dict(checks=20)
                  ):
    """
    Match keypoint descriptors according to a brute force ('bf') or fast
    approximation ('flann') method. Returns the matches.
    """

    support_match = ['bf', 'flann']

    if matchAlgo not in support_match:
        errmsg = 'Matching algorithm {} not supported ' \
                 '(must be one of {})'.format(
                  matchAlgo, ', '.join(support_match))
        raise ValueError(errmsg)
    elif matchAlgo == 'bf':
        # bf = cv2.BFMatcher(cv2.NORM_L2) #cv2.NORM_HAMMING)
        # TODO: decide whether to allow norm to be chosen by user or make
        # dependent on algorithm
        matcher = cv2.BFMatcher(bfNorm)
    elif matchAlgo == 'flann':
        # FLANN is an optimized matcher (versus brute force), so should play
        # with this as well. Maybe brute force is better if we go with multiple
        # subsamples to register image (instead of entire image)
        matcher = cv2.FlannBasedMatcher(flannIndexParams, flannSearchParams)
    else:
        print 'should not have gotten here!'
        raise Exception('wow')

    # The actual matching computation
    matches = matcher.knnMatch(des1, des2, k=2)

    # Return the keypoints and matches as a tuple
    return matches


def feature_find(img1, detectAlgo=cv2.SIFT()):
    """
    Find features in imags img1. Returns the keypoints and descriptors.
    """

    support_detect = ['sift', 'surf', 'orb']

    # The actual feature detection computations
    kp1, des1 = detectAlgo.detectAndCompute(img1, None)

    # Return the keypoints and matches as a tuple
    return (kp1, des1)


def gray_to_3(img):
    '''Return a BRG array ([x,y,3]) representation of a grayscale array'''
    if len(img.shape) > 2:
        errmsg = 'input image is already multichannel'
        raise ValueError(errmsg)
    elif len(img.shape) < 2:
        errmsg = 'input image is not 2-dimensional'
        raise ValueError(errmsg)
    else:
        #Stack the grayscale matrix in each of three channels
        return np.dstack((img, img, img))


def gray(img):
    # Convert to grayscale as the average of three color channels

    # Pre allocating the memory helps improve by ~200 ms
    im_out = np.empty(img.shape[:2], dtype=np.uint16)

    # Similarly a float multiply is cheaper than a float divide
    onethird = 1./3.

    # For grayscale, we want to return integers (or do we?)
    return np.uint8(np.sum(img, axis=-1, out=im_out)*onethird)


def gray3(img):
    # Convenience function
    g = gray(img)
    return gray_to_3(g)


def img_cdf(img, return_inverse=False):
    """
    Calculate the CDF of an image using quicksort. Optional returns from
    np.unique provide the histogram counts and indices for reconstruction of
    the image. The CDF is returned from the normalized cumulative sum of the
    counts.
    """

    uniq = np.unique(img, return_counts=True, return_inverse=return_inverse)
    c = uniq[0]
    f = uniq[-1]
    f = np.cumsum(f).astype(np.float_)
    f /= f[-1]

    # Return the cdf as a tuple of the sorted array and normalized index
    thereturn = (c, f)

    if return_inverse:
        # Add the reverse indices to the return value, which if present are in
        # position 1.
        thereturn += (uniq[1], )

    return thereturn  #(c, f, i)


def neighborhood_cdf(img, channel=None, gr=True, kernel=None):
    """
    Calculate the CDF of an image with additional weighting for points in
    each pixel's neighborhood--with any luck this allows for each pixel to
    be ranked with (very few) ties.

    Create a new matrix using the reference pixels as base and add a fraction
    of intensity level depending on the intensities of the surrounding pixels.

    The default kernel (if not provided) uses 4 surrounding pixels of equal
    weights:

         |1|
       |1|*|1|
         |1|

    """

    # convert the image to uint16 so we can do some fast addition on it
    # if a color channel is not provided convert to grayscale first
    if channel:
        img = np.uint32(img[:, :, channel])
    elif gr:
        img = np.uint32(gray(img))

    if not kernel:
        # If not provided, use a default neighborhood kernel for convolution.
        # Should the center pixel be included? Shouldn't matter since we're adding
        # to the base image.
        # The old default kernel was bigger and took an extra 500-600 ms
        #  kernel0 = np.array([
        #      [0,0,1,0,0],
        #      [0,1,1,1,0],
        #      [1,1,1,1,1],
        #      [0,1,1,1,0],
        #      [0,0,1,0,0]])
        kernel = np.array([
            [0,1,0],
            [1,0,1],
            [0,1,0]])
    elif not isinstance(kernel, np.ndarray):
        # Try to convert the provided kernel if it isn't an array
        kernel = np.array(kernel)
        # If the kernel is not numeric, or if the list is not rectangular, the 
        # dtype will not be integer or float (probably string or object)
        if kernel.dtype not in ['int64', 'float64']:
            errmsg = "The kernel must be a real-valued numeric numpy " +\
                     "array or convertible into such."
            raise ValueError(errmsg)

    weights = np.empty(img.shape, dtype=np.uint32)
    # Compute the sum of the pixels defined by the kernel
    convolve(img, kernel, output = weights)

    # Add weights to the original image (times 8 * 256 = 2048). 
    # --> 256 = 2**8 = number of intensity levels in 8-bit image
    # -->   8 = relative factor of image to neighbor offset.
    #           Needs to be >4 since the convolution is the sum of four; we
    #           use 8 since this just becomes a cheap bit shift operation.
    # TODO: Now that we can pass a custom kernel the appropriate factor should
    # be calculated instead of hardcoded. Seems unlikely anyone will actually
    # use another kernel at this time (7/5/16: CMP).
    # This yields the same results as creating fractional intensities in
    # weights and adding to the original image but keeps the math in integer
    # space.
    # End result is a matrix whose relative values are the same as the original
    # image except for an offset at each pixel dependent on its neighbors.
    weights += img * 2048

    # img_cdf doesn't care that the values are outside of [0,255], so its
    # cdf is the same modulo the above scaling factor.
    # And yeah, I'm being loose with the word 'modulo'.
    return img_cdf(weights, return_inverse=True), weights


def get_intensity(cdf, intensities, vals):
    """
    Generator function to iteratively return intensities from one cdf given
    another cdf ('vals')
    """
    idx = 0
    val = vals[idx]
    for c, i in zip(cdf, intensities):
        if c >= val:
            idx += 1
            val = vals[idx]
            yield i


def hist_match(ref, img, gr=True, channel=0):
    """
    Adjust the intensities of img to match the histogram of ref using their
    cdf's.
    """

    indices = np.empty(img.shape[0]*img.shape[1], dtype = np.int32)

    # ref_intensities and ref_cdf are the quantized intensities
    if gr:
        ref_intensities, ref_cdf = img_cdf(gray(ref))
        #(im_intensities, im_cdf), im12 = neighborhood_cdf(gray(img))
        (im_intensities, im_cdf,  indices), im12 = neighborhood_cdf(img, gr=gr)
    else:
        ref_intensities, ref_cdf = img_cdf(ref[:, :, channel])
        (im_intensities, im_cdf, indices), im12 = neighborhood_cdf(img[:,:,channel])

    g = np.interp(im_cdf, ref_cdf, ref_intensities)

    adjusted_img = g[indices].reshape(img.shape[:2])
    return adjusted_img


# def calculate_img_offset(img1, img2, **matchops):
#     '''
#     '''
# 
#     kp1, des1, kp2, des2, matches = feature_match(img1, img2, **matchops)
#             #cv2.SIFT(), 'flann')  # , cv2.ORB())
#     xx = []
#     yy = []
#     dd = []
#     tt = []
#     for mat in matches:
# 
#         # Get the matching keypoints for each of the images
#         img1_idx = mat.queryIdx
#         img2_idx = mat.trainIdx
#         # x - columns
#         # y - rows
#         (x1, y1) = kp1[img1_idx].pt
#         (x2, y2) = kp2[img2_idx].pt
#         t1 = kp1[img1_idx].angle
#         t2 = kp2[img2_idx].angle
# 
#         dx = x2-x1
#         dy = y2-y1
#         dt = t2-t1
#         xx.append(dx)
#         yy.append(dy)
#         tt.append(dt)
#         # Compute the distance between matching keypoints
#         d = (dx**2 + dy**2)**0.5
#         dd.append(d)
# 
# #    pl.hist(dd)
# #    pl.show()
# 
#     print 'dx mean = {}, dx sd = {}'.format(np.mean(xx), np.std(xx))
#     print 'dy mean = {}, dy sd = {}'.format(np.mean(yy), np.std(yy))
#     print 'dist mean = {}, dist sd = {}'.format(np.mean(dd), np.std(dd))
#     print 'dt mean = {}, dt sd = {}'.format(np.mean(tt), np.std(tt))
# 
#     xx = np.array(xx)
#     yy = np.array(yy)
#     dd = np.array(dd)
#     tt = np.array(tt)
#     xxint = xx.astype(int)
#     yyint = yy.astype(int)
#     ttint = tt.astype(int)
# 
#     xoffset = (np.bincount(xxint-xxint.min()).argmax() + xxint.min())
#     yoffset = (np.bincount(yyint-yyint.min()).argmax() + yyint.min())
#     toffset = (np.bincount(ttint-ttint.min()).argmax() + ttint.min())
# 
#     print "x_peak = {0}".format(xoffset)
#     print "y_peak = {0}".format(yoffset)
#     print "t_peak = {0}".format(toffset)
# 
#     return xoffset, yoffset, toffset, xx, yy, dd, tt


def calculate_img_offset(ref, img, histmatch=False, histdir='left',
        detectAlgo=cv2.SIFT(), **matchops):
    ''
    ''

    img1 = ref
    img2 = img

    if histmatch:
        if histdir == 'left':
            img1 = gray(img1)
            img2 = np.uint8(hist_match(img1, img2), gr=True)
        else:
            img1 = np.uint8(hist_match(img2, img1), gr=True)
            img2 = gray(img2)

    print f,
    kp1, des1 = feature_find(img1, detectAlgo=detectAlgo)
    kp2, des2 = feature_find(img2, detectAlgo=detectAlgo)

    matches = match_des(des1, des2, **matchops)
    #from opencv tutorial: Feature Matching + Homography to find Objects
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)

    if len(good)>10: #MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        #don't really need a homography, but getAffineTransform doesn't seem
        #to support RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None
    print len(matches), len(good), sum(matchesMask)

    ransacpoints = np.asarray(good)[np.array(matchesMask, dtype=np.bool)]
    srcpts = []
    dstpts = []

    for mat in ransacpoints: #[matchesMask]:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # Compute the distance between matching keypoints
        srcpts.append([x1, y1])
        dstpts.append([x2, y2])

    srcarray = np.array(srcpts,dtype=np.float32).reshape(1,len(srcpts),2)
    dstarray = np.array(dstpts,dtype=np.float32).reshape(1,len(dstpts),2)
    print srcarray, dstarray

    rt = cv2.estimateRigidTransform(srcarray, dstarray, False)
    print rt
    if not rt is None:
        pl.imshow(img2, cmap='gray')
        pl.show()
        print 'dx = ', rt[0,2],
        print 'dy = ', rt[1,2],
        print 'dt = ', asin(rt[1,0])


def calculate_img_offset_batch(ref, flist, histmatch = False,
        detectAlgo=cv2.SIFT(), **matchops):
    '''
    '''

    img1 = ref
    kp1, des1 = feature_find(gray(img1), detectAlgo=detectAlgo)

    for f in flist:
        img2 = loadRAW(f)
        #img2 = gray(img2)
        if histmatch:
            img2 = np.uint8(hist_match(img1, img2), gr=True)

        print f,
        kp2, des2 = feature_find(img2, detectAlgo=detectAlgo)

        matches = match_des(des1, des2, **matchops)
        #return matches
        #break
        #from opencv tutorial: Feature Matching + Homography to find Objects
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)

        if len(good)>10: #MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            #don't really need a homography, but getAffineTransform doesn't seem
            #to support RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            #print matchesMask
            # h,w = img1.shape[:2]
            # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # dst = cv2.perspectiveTransform(pts,M)

            # img6 = cv2.polylines(img2,[np.int32(dst)],True,255,3) #, cv2.LINE_AA)
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None
        #return dst
        print len(matches), len(good), sum(matchesMask)

        ransacpoints = np.asarray(good)[np.array(matchesMask, dtype=np.bool)]
        srcpts = []
        dstpts = []

        for mat in ransacpoints: #[matchesMask]:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
            # x - columns
            # y - rows
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            # Compute the distance between matching keypoints
            srcpts.append([x1, y1])
            dstpts.append([x2, y2])

        srcarray = np.array(srcpts,dtype=np.float32).reshape(1,len(srcpts),2)
        dstarray = np.array(dstpts,dtype=np.float32).reshape(1,len(dstpts),2)
        print srcarray, dstarray

        rt = cv2.estimateRigidTransform(srcarray, dstarray, False)
        print rt
        if not rt is None:
            pl.imshow(img2, cmap='gray')
            pl.show()
            print 'dx = ', rt[0,2],
            print 'dy = ', rt[1,2],
            print 'dt = ', asin(rt[1,0])
    return rt

# def calculate_img_offset_batch(ref, flist, histmatch = False,
#         detectAlgo=cv2.SIFT(), **matchops):
#     '''
#     '''
# 
#     img1 = ref
#     kp1, des1 = feature_find(gray(img1), detectAlgo=detectAlgo)
# 
#     for f in flist:
#         img2 = loadRAW(f)
#         #img2 = gray(img2)
#         if histmatch:
#             img2 = np.uint8(hist_match(img1, img2))
# 
#         print f,
#         kp2, des2 = feature_find(img2, detectAlgo=detectAlgo)
# 
#         matches = match_des(des1, des2, **matchops)
#         #return matches
#         #break
#         #from opencv tutorial: Feature Matching + Homography to find Objects
#         good = []
#         for m,n in matches:
#             if m.distance < 0.8*n.distance:
#                 good.append(m)
# 
#         if len(good)>10: #MIN_MATCH_COUNT:
#             src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#             dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
# 
#             #don't really need a homography, but getAffineTransform doesn't seem
#             #to support RANSAC
#             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#             matchesMask = mask.ravel().tolist()
#             #print matchesMask
#             # h,w = img1.shape[:2]
#             # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#             # dst = cv2.perspectiveTransform(pts,M)
# 
#             # img6 = cv2.polylines(img2,[np.int32(dst)],True,255,3) #, cv2.LINE_AA)
#         else:
#             print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#             matchesMask = None
#         #return dst
#         print len(matches), len(good), sum(matchesMask)
# 
#         ransacpoints = np.asarray(good)[np.array(matchesMask, dtype=np.bool)]
#         xx = []
#         yy = []
#         dd = []
#         tt = []
#         for mat in ransacpoints: #[matchesMask]:
# 
#             # Get the matching keypoints for each of the images
#             img1_idx = mat.queryIdx
#             img2_idx = mat.trainIdx
#             # x - columns
#             # y - rows
#             (x1, y1) = kp1[img1_idx].pt
#             (x2, y2) = kp2[img2_idx].pt
#             t1 = kp1[img1_idx].angle
#             t2 = kp2[img2_idx].angle
# 
#             dx = x2-x1
#             dy = y2-y1
#             dt = t2-t1
#             xx.append(dx)
#             yy.append(dy)
#             tt.append(dt)
#             # Compute the distance between matching keypoints
#             d = (dx**2 + dy**2)**0.5
#             dd.append(d)
# 
#     #    pl.hist(dd)
#     #    pl.show()
# 
# #         print 'dx mean = {}, dx sd = {}'.format(np.mean(xx), np.std(xx))
# #         print 'dy mean = {}, dy sd = {}'.format(np.mean(yy), np.std(yy))
# #         print 'dist mean = {}, dist sd = {}'.format(np.mean(dd), np.std(dd))
# #         print 'dt mean = {}, dt sd = {}'.format(np.mean(tt), np.std(tt))
# 
#         xx = np.array(xx)
#         yy = np.array(yy)
#         dd = np.array(dd)
#         tt = np.array(tt)
#         xxint = xx.astype(int)
#         yyint = yy.astype(int)
#         ttint = tt.astype(int)
# 
#         xoffset = (np.bincount(xxint-xxint.min()).argmax() + xxint.min())
#         yoffset = (np.bincount(yyint-yyint.min()).argmax() + yyint.min())
#         toffset = (np.bincount(ttint-ttint.min()).argmax() + ttint.min())
# 
#         print "x_peak = {0}; ".format(xoffset),
#         print "y_peak = {0}; ".format(yoffset),
#         print "t_peak = {0}".format(toffset)
# 
#     #return None
#     return xoffset, yoffset, toffset, xx, yy, dd, tt


if __name__ == '__main__':
    thedir = '/projects/projects/project-urban_lightscape/datamarts/urban_lightscape/2013/10/26/23.53.29/'
    fname = os.getenv('cuipimg') + 'temp__2014-09-29-125314-29546.raw'
    flist = os.listdir(thedir)
    #ref = loadRAW(thedir + flist[0])   # stable results! (but correct?)
    ref = loadRAW(fname)                # unstable results! (and definitely wrong)
    pl.imshow(gray(ref), cmap='gray')
    pl.show()
    flist = [thedir + f for f in flist]
    calculate_img_offset_batch(ref, flist, histmatch = False, detectAlgo=cv2.ORB())

