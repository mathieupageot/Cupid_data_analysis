import numpy as np
import matplotlib.pyplot as plt
import get_data
from landaupy import langauss
import gaussian_fit_light


def gauss(x, h, a, x0, sigma):
    return h + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def get_heat(path,filename,p):
    peaks = get_data.ntd_array(path + filename)
    amp_stab = np.load(path + 'amp_stab.npy')
    E = p(amp_stab)
    correlation = peaks[:, 5]
    TV = peaks[:, 8]
    riset = peaks[:, 11]
    decayt = peaks[:, 12]
    Sm = peaks[:, 9] / amp_stab
    print((peaks[0,0]-peaks[-1,0])/5000/60/60)
    return E,correlation,TV,riset,decayt,Sm

def f(x, a, b, c):
    return a * x ** b + c
def f_tv(x, a, b, c):
    return np.exp((x - a) / b) * c

def plot_TV(E,TV,TV_cut,para_TV):
    fig1, axs1 = plt.subplots(1)
    axs1.scatter(E, TV, s=0.1)
    axs1.scatter(E[TV_cut], TV[TV_cut], s=0.1)
    xp = np.linspace(0, 80000)
    axs1.plot(xp, f(xp, *para_TV))
    axs1.set_xlabel('Pulse energy in keV')
    axs1.set_ylim(0,TV.max())
    axs1.set_ylabel('TV')
    axs1.set_title('Energy vs TV')


def plot_rise(E,riset):
    fig5, ax5 = plt.subplots(1)
    ax5.scatter(E, riset, s=0.1)
    ax5.set_ylim(-0.02,0.06)
    ax5.set_xlabel('Pulse energy in keV')
    ax5.set_ylabel('Rise time in s')
    ax5.set_title('Energy vs Rise Time')
def plot_decay(E,decayt):
    fig, ax = plt.subplots(1)
    ax.scatter(E, decayt, s=0.1)
    ax.set_ylim(0,0.1)
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylabel('Decay time in s')
    ax.set_title('Energy vs Decay Time')
def plot_corr(E,correlation,para):
    fig6, axs6 = plt.subplots()
    axs6.scatter(E, correlation, s=0.1)
    axs6.set_ylim(0.98, 1.)
    xcor = np.linspace(E.min(), E.max(), 1000)
    axs6.plot(xcor, f(xcor,*para), c='r', label='correlation cut')
    axs6.legend()
    axs6.set_title('Correlation vs Pulse energy raw data heat channel')
    axs6.set_ylabel('Correlation normalized')
    axs6.set_xlabel('Pulse energy in arbitrary unit')
def hist_heat_cut(E,TV_cut,correl_cut):
    fig2, axs2 = plt.subplots(3)
    n2, bins2 = np.histogram(E[E < 6000], 3000)
    center2 = (bins2[:-1] + bins2[1:]) / 2
    axs2[0].plot(center2, n2, linewidth=.5, ds='steps-mid', label='no cut')
    axs2[0].set_ylabel('Counts/2 keV')

    amp_TV = E[TV_cut]
    nTV, binsTV = np.histogram(amp_TV[amp_TV < 6000], 3000)
    centerTV = (binsTV[:-1] + binsTV[1:]) / 2
    axs2[1].plot(centerTV, nTV, linewidth=.5, ds='steps-mid', label='TV cut')
    axs2[1].set_ylabel('Counts/2 keV')

    amp_corr = E[correl_cut]
    ncorr, binscorr = np.histogram(amp_corr[amp_corr < 6000], 3000)
    centercorr = (binscorr[:-1] + binscorr[1:]) / 2
    axs2[2].plot(centercorr, ncorr, linewidth=.5, ds='steps-mid', label='Correlation cut')
    axs2[2].set_xlabel('Pulse energy in keV')
    axs2[2].set_ylabel('Counts/2 keV')
    for axis in axs2:
        axis.legend()
def import_light(path ,filename_trigheat,filename_light,sel):
    peaksl = get_data.ntd_array(path + filename_trigheat)
    ampl = peaksl[sel, 2]
    peakslt = get_data.ntd_array(path + filename_light)
    corlt = peakslt[:, 5]
    sellt = corlt > 0.99
    amplt = peakslt[sellt, 2]
    return amplt,ampl

def plot_Sm(E,Sm):
    fig10, ax10 = plt.subplots()
    ax10.scatter(E, Sm, s=0.1)
    ax10.set_xlabel('Amplitude in keV')
    ax10.set_ylabel('S$_m$/m')
    ax10.set_title('The Fitted Amplitude vs Energy')
    ymean=np.mean(Sm)
    yvar=np.var(Sm)
    ax10.set_ylim(ymean-100*yvar,ymean+100*yvar)
def hist_light_fit(amplt,filenum):
    fig4, axs4 = plt.subplots(1)
    popt, pcov, hist, bin_centers = gaussian_fit_light.binned_fit_langauss(amplt[amplt > 3400], bins='auto',
                                                                           nan='remove')
    axs4.plot(bin_centers, hist, linewidth=.5, ds='steps-mid')
    x_plot = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    axs4.plot(x_plot, langauss.pdf(x_plot, *popt))
    axs4.set_xlabel("Light amplitude in arbitrary unit")
    axs4.set_ylabel("Counts/" + str(int((bin_centers[-1] - bin_centers[0]) / len(bin_centers)) + 1) + ' arbitrary units')
    axs4.set_title('Landeau fit of the light channel')

    if filenum == 2:
        amplt_fit = amplt * 100 / popt[0]
        ampl_fit = ampl * 100 / popt[0]
        print('keV/ADU for light: ' + str(100 / popt[0]) + ' & error: ' + str(np.sqrt(pcov[0, 0])* 100 / popt[0]**2))
    elif filenum == 3:
        amplt_fit = amplt * 260 / popt[0]
        ampl_fit = ampl * 260 / popt[0]
        print('keV/ADU for light: ' + str(260 / popt[0]) + ' & error: ' + str(np.sqrt(pcov[0, 0])* 620 / popt[0]**2))
    else:
        amplt_fit = amplt * 100 / popt[0]
        ampl_fit = ampl * 100 / popt[0]
        print('keV/ADU for light: ' + str(100 / popt[0]) + ' & error: ' + str(np.sqrt(pcov[0, 0])* 100 / popt[0]**2))
    return amplt_fit, ampl_fit

def hist_light(amplt_fit):
    fig7, ax7 = plt.subplots()
    n, bins = np.histogram(amplt_fit[amplt_fit < 6000], 500)
    center = (bins[:-1] + bins[1:]) / 2
    ax7.plot(center, n, linewidth=.5, ds='steps-mid', label='Calibrated')
    ax7.set_xlabel('Light Channel Energy in keV')
    ax7.set_ylabel('Counts/12 keV')


def plot_light(E_sel,ampl_fit):
    fig3, ax3 = plt.subplots(1)  # heatvslight
    ax3.scatter(E_sel, ampl_fit, s=0.1)
    ax3.set_ylabel('Light amplitude in keV')
    ax3.set_xlabel('Heat amplitude in keV')
    ax3.set_title('Heat amplitude VS Light amplitude for alpha discrimination')
    ax3.set_ylim(-5,5)
def plot_LY(E_sel,LY):
    fig11, ax11 = plt.subplots()
    ax11.scatter(E_sel,LY,s=0.1)
    ax11.set_ylabel('LY ADU')
    ax11.set_xlabel('Heat amplitude in keV')
    ax11.set_title('Heat Energy vs LY')
    ymean=np.mean(LY)
    yvar=np.var(LY)
    ax11.set_ylim(ymean-yvar,ymean+yvar)

if __name__ == '__main__':
    path, filename, filename_light, filename_trigheat, p= get_data.get_path()
    E, correlation, TV, riset, decayt, Sm = get_heat(path, filename, p)
    try:
        para_corr = np.load(path+filename.strip(".ntp")+'_'+'correlation'+".npy")
    except FileNotFoundError:
        print('No correlation cut found')
        para_corr = np.array([-1,-1,0.80])
    correl_cut = correlation > f(E,*para_corr)
    rise_cut = riset < 0.25
    sel = np.logical_and(correl_cut, rise_cut)
    try:
        para_TV = np.load(path+filename.strip(".ntp")+'_'+'TV'+".npy")
    except FileNotFoundError:
        print('No TV cut found')
        para_TV = np.array([4,4,1])
    TV_cut = f(E, *para_TV) > TV
    E_sel = E[sel]
    plot_TV(E,TV,TV_cut,para_TV)
    plot_rise(E, riset)
    plot_decay(E, decayt)
    plot_corr(E, correlation,para_corr)
    plot_Sm(E, Sm)
    hist_heat_cut(E, TV_cut, correl_cut)
    if filename_light != 0 :
        amplt, ampl = import_light(path, filename_trigheat, filename_light, sel)
        amplt_fit, ampl_fit = hist_light_fit(amplt,filenum)
        LY = ampl_fit / E_sel
        plot_light(E_sel, ampl_fit)
        plot_LY(E_sel, LY)

    plt.show()