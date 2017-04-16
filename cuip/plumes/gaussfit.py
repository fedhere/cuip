import numpy as np
from scipy.optimize import curve_fit

def gauss(x, *p):
    A, mu, sig = p
    return A * np.exp(-(x-mu)**2 / (2. * sig**2))

def gaussfit(datay, datax, A, mu, sig):
    coeff, var = curve_fit(gauss, datax, datay, p0=[A, mu, sig])
    return gauss(datax, *coeff), coeff

def lorentzian(x, *p):
	A, x0 = p
	return A / ((x-x0)**2 + A**2) / np.pi

def lorentzfit(datay, datax, A, x0):
	coeff, var = curve_fit(lorentzian, datax, datay, p0 = [A, x0])
	return lorentzian(datax, *coeff), coeff#, coeff[0], coeff[1]


#Lorentzian first guess
#a_guess = 1 / (np.pi * max(ydata))
#x0_guess = sum(xdata * ydata) / sum(ydata)

