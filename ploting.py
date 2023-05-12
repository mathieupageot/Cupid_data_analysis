import numpy as np
import matplotlib.pyplot as plt
import get_data
from landaupy import langauss
import gaussian_fit_light


def gauss(x, h, a, x0, sigma):
    return h + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


fig4, axs4 = plt.subplots(1)
fig2, axs2 = plt.subplots(3)  # histo bf after
fig7, ax7 = plt.subplots()
fig, axs = plt.subplots(1)  # decaytime plot
fig6, axs6 = plt.subplots(1)  # correlation plot
fig5, axs5 = plt.subplots(1)  # risetime plot

fig1, axs1 = plt.subplots(1)  # TV plot
fig3, axs3 = plt.subplots(1)  # heatvslight
path, filename, filename_light, filename_trigheat, filenum = get_data.get_path()
peaks = get_data.ntd_array(path + filename)
amp_stab = np.load(path + 'amp_stab.npy')

if filenum == 2:
    z = [0.54076489, 6.23910689]
if filenum == 3:
    z = [2.47881842, -10.57679343]
else:
    z = [1, 0]


# z= [ 1.17286046, 78.40799081]
def f_corr(x):
    corr_end = 0.9998
    corr_0 = 0.994
    return -1 / (x + 1 / (corr_end - corr_0)) + corr_end


def f_tv(x, a, b, c):
    return np.exp((x - a) / b) * c


p = np.poly1d(z)
amp_stab_fit = p(amp_stab)
correlation = peaks[:, 5]
t = peaks[:, 0] / 5000
TV = peaks[:, 8]  # parabolic cut
Riset = peaks[:, 11]
decayt = peaks[:, 12]

correl_cut = correlation > f_corr(amp_stab_fit)
Rise_cut = Riset < 0.25

good = np.logical_and(correl_cut, Rise_cut)

axs1.scatter(amp_stab_fit, TV, s=0.1)

para = [36578.7450938, 8284.52785707, 101708.98640136]
xp = np.linspace(0, 80000)
TV_cut = f_tv(amp_stab_fit, *para) < TV

axs1.scatter(amp_stab_fit[TV_cut], TV[TV_cut], s=0.1)

axs1.plot(xp, f_tv(xp, *para))
axs1.set_xlabel('Pulse energy in keV')
axs1.set_ylabel('TV')
axs1.set_title('Energy vs TV')

axs5.scatter(amp_stab_fit, Riset, s=0.1)
axs5.hlines(Rise_cut, amp_stab_fit.min(), amp_stab_fit.max(), colors='r', label='Rise time cut')
axs5.set_xlabel('Pulse energy in keV')
axs5.set_ylabel('Rise time in s')
axs5.set_title('Energy vs Rise Time')

axs.scatter(amp_stab_fit, decayt, s=0.1)
axs.set_xlabel('Pulse energy in keV')
axs.set_ylabel('Decay time in s')
axs.set_title('Energy vs Decay Time')

axs6.scatter(amp_stab_fit, correlation, s=0.1)
axs6.set_ylim(0.90, 1.01)
xcor = np.linspace(amp_stab_fit.min(), amp_stab_fit.max(), 1000)
axs6.plot(xcor, f_corr(xcor), c='r', label='correlation cut')
axs6.legend()
axs6.set_title('Correlation vs Pulse energy raw data heat channel')
axs6.set_ylabel('Correlation normalized')
axs6.set_xlabel('Pulse energy in arbitrary unit')

n2, bins2 = np.histogram(amp_stab_fit[amp_stab_fit < 6000], 3000)
center2 = (bins2[:-1] + bins2[1:]) / 2
axs2[0].plot(center2, n2, linewidth=.5, ds='steps-mid', label='no cut')
axs2[0].set_ylabel('Counts/2 keV')

amp_TV = amp_stab_fit[TV_cut]
nTV, binsTV = np.histogram(amp_TV[amp_TV < 6000], 3000)
centerTV = (binsTV[:-1] + binsTV[1:]) / 2
axs2[1].plot(centerTV, nTV, linewidth=.5, ds='steps-mid', label='TV cut')

axs2[1].set_ylabel('Counts/2 keV')
amp_corr = amp_stab_fit[correl_cut]
ncorr, binscorr = np.histogram(amp_corr[amp_corr < 6000], 3000)
centercorr = (binscorr[:-1] + binscorr[1:]) / 2
axs2[2].plot(centercorr, ncorr, linewidth=.5, ds='steps-mid', label='Correlation cut')
axs2[2].set_xlabel('Pulse energy in keV')
axs2[2].set_ylabel('Counts/2 keV')
for axis in axs2:
    axis.legend()

peaksl = get_data.ntd_array(path + filename_trigheat)

tl = peaksl[good, 0]
ampl = peaksl[good, 2]

correlationl = peaksl[good, 5]

baselinel = peaksl[good, 3]
good2 = correlationl > 0.99

peakslt = get_data.ntd_array(path + filename_light)
corlt = peakslt[:, 5]
goodlt = corlt > 0.99
amplt = peakslt[goodlt, 2]

popt, pcov, hist, bin_centers = gaussian_fit_light.binned_fit_langauss(amplt[amplt > 3400], bins='auto', nan='remove')
axs4.plot(bin_centers, hist, linewidth=.5, ds='steps-mid')
x_plot = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
axs4.plot(x_plot, langauss.pdf(x_plot, *popt))
axs4.set_xlabel("Light amplitude in arbitrary unit")
axs4.set_ylabel("Counts/" + str(int((bin_centers[-1] - bin_centers[0]) / len(bin_centers)) + 1) + ' arbitrary units')
axs4.set_title('Landeau fit of the light channel')

if filenum == 2:
    amplt_fit = amplt * 100 / popt[0]
    print(100 / popt[0])
if filenum == 3:
    amplt_fit = amplt * 260 / popt[0]
    print(260 / popt[0])
else:
    amplt_fit = amplt * 100 / popt[0]
    print(100 / popt[0])

n, bins = np.histogram(amplt_fit[amplt_fit < 6000], 500)
center = (bins[:-1] + bins[1:]) / 2
ax7.plot(center, n, linewidth=.5, ds='steps-mid', label='Calibrated')
ax7.set_xlabel('Light Channel Energy in keV')
ax7.set_ylabel('Counts/12 keV')

pts = axs3.scatter(amp_stab_fit[good], ampl * 100 / popt[0], s=0.1)
axs3.set_ylabel('Light amplitude in keV')
axs3.set_xlabel('Heat amplitude in keV')
axs3.set_title('Heat amplitude VS Light amplitude for alpha discrimination')
fig10, ax10 = plt.subplots()
ax10.scatter(amp_stab_fit, peaks[:, 9] / amp_stab, s=0.1)
ax10.set_xlabel('Amplitude in keV')
ax10.set_ylabel('S$_m$/m')
plt.show()
