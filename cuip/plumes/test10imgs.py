import glob
import os
import sys
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

PLUMES = os.getenv('PLUMES_DATA')
sys.path.append(PLUMES)
import gaussfit as gf



img_med = np.zeros((20, 1300, 4096))
img_mdif = np.zeros((20, 1300, 4096))

for i,f in enumerate(sorted(glob.glob(os.path.join(PLUMES,'outputs/tmp_01[45][0-9]_median.npy')))):
    print(f)
    img_med[i] = np.load(f)[200:1500,:,:].mean(-1)

for i,f in enumerate(sorted(glob.glob(os.path.join(PLUMES, 'outputs/tmp_01[45][0-9]_mindif.npy')))):
    print(f)
    img_mdif[i] = np.load(f)[200:1500,:,:].mean(-1)


BINS = 64   #int(round(2*len(img_med[0].flatten())**(1./3.)))
y_med = np.zeros((20, BINS))
y_mdif = np.zeros((20, BINS))
x = np.zeros(BINS)
patches = np.zeros(BINS)


for i in range(20):
    y_med[i], x, patches = plt.hist(img_med[i].flatten(), bins = BINS, log = True)
    y_mdif[i] = plt.hist(img_mdif[i].flatten(), bins = BINS, log = True)[0]
    
ymed_log = np.log10(y_med)
ymdif_log = np.log10(y_mdif)

ymed_log[ymed_log == -np.inf] = 0.0
ymdif_log[ymdif_log == -np.inf] = 0.0

bins_cen = x[:-1] + 0.5*(x[1]-x[0])

#select images with clear plumes
plumes_med = ymed_log[8:12]
plumes_mdif = ymdif_log[8:12]
ymed_fit = np.zeros((len(plumes_med), BINS))
ymdif_fit = np.zeros((len(plumes_mdif), BINS))
mu_med = np.zeros((len(plumes_med)))
mu_mdif = np.zeros((len(plumes_med)))
sigma_med = np.zeros((len(plumes_mdif)))
sigma_mdif = np.zeros((len(plumes_mdif)))


for i in range(len(plumes_med)):
    mu_med[i] = sum(bins_cen*plumes_med[i])/sum(plumes_med[i])
    sigma_med[i] = np.sqrt(sum(plumes_med[i]))


for i in range(len(plumes_med)):
    ymed_fit[i] = gf.gaussfit(plumes_med[i], bins_cen, plumes_med[i].max(), mu_med[i], sigma_med[i])
    

#for i in range(len(plumes_mdif)):
 #   mu_mdif[i] = sum(bins_cen*plumes_mdif[i])/sum(plumes_mdif[i])
 #   sigma_mdif[i] = np.sqrt(sum(plumes_mdif[i]))


#for i in range(len(plumes_mdif)):
#    ymed_fit[i] = gf.gaussfit(plumes_mdif[i], bins_cen, plumes_mdif[i].max(), mu_mdif[i], sigma_mdif[i])
    


plt.ion()
plt.figure()
plt.subplot(111)

for i in range(len(plumes_med)):
    plt.bar(bins_cen, plumes_med[i], alpha = 0.75)
    plt.plot(bins_cen, ymed_fit[i], 'r-')
    plt.draw()
    raw_input()

#plt.figure(2)
#plt.subplot(111)

#for i in range(len(plumes_mdif)):
 #   plt.bar(bins_cen, plumes_mdif[i], alpha = 0.75)
  #  plt.plot(bins_cen, ymed_fit[i], 'r-')
   # plt.draw()
    #raw_input()



