import os
import numpy as np
import matplotlib.pyplot as plt

plumes = os.getenv('PLUMES_DATA') #plumes project, input files
npy_path = os.path.join(plumes, 'outputs') #dir with np arrays for anlysis
root = os.getenv('CUIP') # plumes repo



#select images with clear plumes
plumes_med = ymed_log[8:12]
plumes_mdif = ymdif_log[8:12]

#empty vectors to store the gaussian fitting. 
ymed_fit = np.zeros((len(plumes_med), BINS))
ymdif_fit = np.zeros((len(plumes_mdif), BINS))

#initializing mean and sigma
mu_med = np.zeros((len(plumes_med)))
mu_mdif = np.zeros((len(plumes_med)))
sigma_med = np.zeros((len(plumes_mdif)))
sigma_mdif = np.zeros((len(plumes_mdif)))


###MEDIAN.NPY fitting

#calculate mean and sigma for each image dist. for first guess in curve fitting
for i in range(len(plumes_med)):
    mu_med[i] = sum(bins_cen*plumes_med[i])/sum(plumes_med[i])
    sigma_med[i] = np.sqrt(sum(plumes_med[i]))

#fitting for each image dist
for i in range(len(plumes_med)):
    ymed_fit[i] = gf.gaussfit(plumes_med[i], bins_cen, plumes_med[i].max(), mu_med[i], sigma_med[i])

plt.ion()
plt.figure()
plt.subplot(111)

#plotting of the histograms with respective fitted curve
for i in range(len(plumes_med)):
    plt.bar(bins_cen, plumes_med[i], alpha = 0.75)
    plt.plot(bins_cen, ymed_fit[i], 'r-')
    plt.draw()
    raw_input()
'''
### FITTING AND PLOTTING FOR MINDIF.NPY. TO-DO

#for i in range(len(plumes_mdif)):
 #   mu_mdif[i] = sum(bins_cen*plumes_mdif[i])/sum(plumes_mdif[i])
 #   sigma_mdif[i] = np.sqrt(sum(plumes_mdif[i]))


#for i in range(len(plumes_mdif)):
#    ymed_fit[i] = gf.gaussfit(plumes_mdif[i], bins_cen, plumes_mdif[i].max(), mu_mdif[i], sigma_mdif[i])


#plt.figure(2)
#plt.subplot(111)

#for i in range(len(plumes_mdif)):
 #   plt.bar(bins_cen, plumes_mdif[i], alpha = 0.75)
  #  plt.plot(bins_cen, ymed_fit[i], 'r-')
   # plt.draw()
    #raw_input()



