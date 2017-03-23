import numpy as np
from scipy.optimize import curve_fit

def gauss(x, *p):
    A, mu, sig = p
    return A * np.exp(-(x-mu)**2 / (2. * sig**2))

def gaussfit(datay, datax, A, mu, sig):
    coeff, var = curve_fit(gauss, datax, datay, p0=[A, mu, sig])
    return gauss(datax, *coeff)
