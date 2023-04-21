import numpy as np
import matplotlib.pyplot as plt
import lasso_selection
import get_data
from scipy.optimize import curve_fit

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
fig4, axs4= plt.subplots(1)
filename_light='/20211125_00h43.BINLD21.2.ntp'
filename_trigheat='/20211125_00h43.BINLD21.2_trigheat.ntp'
path = '/Users/mp274748/Documents/data_arg/second_set'
peakslt=get_data.ntd_array(path + filename_light)
corlt=peakslt[:,5]
goodlt = corlt > 0.99
amplt=peakslt[goodlt,2]

n, bins,patches = axs4.hist(amplt,200)
bin_center=(bins[:-1]+bins[1:])/2
parameters, covariance = curve_fit(gauss, bin_center, n,p0=[6,75,13000,2000])
x_plot=np.linspace(bin_center.min(),bin_center.max(),1000)
axs4.plot(x_plot,gauss(x_plot,*parameters))


plt.show()
