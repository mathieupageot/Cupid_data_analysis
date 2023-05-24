import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider

import get_data
def lin(x, a, b, c):
    return a * x ** b + c

def plot_corr_cut(amp,para_cut):
    figcorr,axcorr = plt.subplots()
    axcorr.scatter(amp,correlation,s=0.1)
    axcorr.plot(np.sort(amp),lin(np.sort(amp),*para_cut),c='r')
    axcorr.set_ylim(0.98,1)
    axcorr.set_ylabel('Correlation')
    axcorr.set_xlabel('Amplitude in arbitrary unit')


def hist_RAW(amp,nbin):
    n, bins = np.histogram(amp, nbin)
    center = (bins[:-1] + bins[1:]) / 2
    return n, center


def calibr_amp(data_amp,data_E):
    if len(data_E) > 2:
        z, cov = np.polyfit(data_amp, data_E, 1, cov=True)
    else:
        z = np.polyfit(data_amp, data_E, 1)
        cov = np.ones((2, 2))
    return z, cov


def build_line(axcalib,data_amp,data_E,nRAW):
    lcalib = []
    for i in range(len(data_amp)):
        l1, = axcalib.plot([data_E[i],data_E[i]],[0,nRAW.max()], label=str(np.round(data_E[i],1))+' keV',alpha=0.4, linewidth=.5)
        lcalib.append(l1)
        axcalib.legend()
    return lcalib

def plot_fit(data_amp,data_E,p):
    figfit, axfit = plt.subplots()
    axfit.scatter(data_amp,data_E)
    x = np.linspace(min(data_amp) - max(data_amp) * 0.1, max(data_amp) * 1.1, 700)
    y = p(x)
    axfit.plot(x, y, c='r')
    axfit.set_xlabel('Amplitude in arbitrary unit')
    axfit.set_ylabel('Energy in keV')


if __name__ == "__main__":
    path, filename, filename_light, filename_trigheat,data_E,data_amp= get_data.get_path(peak=True)
    peaks = get_data.ntd_array(path+filename)
    amp= np.load(path+'amp_stab.npy')
    correlation = peaks[:,5]
    Riset=peaks[:, 11]
    try:
        para_cut = np.load(path+filename.strip(".ntp")+'_'+'correlation'+".npy")
    except FileNotFoundError:
        para_corr = np.array([-1, -1, 0.80])

    good = np.logical_and(correlation>lin(correlation,*para_cut),amp<2000)
    plot_corr_cut(amp,para_cut)
    amp=amp[good]
    fig, axRAW = plt.subplots()
    fig.subplots_adjust(left=0.25)
    axbin = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    axcalib = axRAW.twiny()




    nbin = 300
    n, center = hist_RAW(amp, nbin)
    z, cov = calibr_amp(data_amp,data_E)
    p = np.poly1d(z)

    plot_fit(data_amp, data_E, p)
    lcalib = build_line(axcalib,data_amp,data_E,n)
    hist, = axRAW.plot(center, n, linewidth=.5, ds='steps-mid')


    def update(val):
        nbin = int(val)
        n, center = hist_RAW(amp, nbin)
        hist.set_xdata(center)
        hist.set_ydata(n)
        for i, line in enumerate(lcalib):
            line.set_ydata([0, n.max()])
        axRAW.set_ylim(0, n.max())
        axRAW.set_ylabel('Number of event/' + str(np.round((p(x2) - p(x1)) / nbin, 1)) + ' keV')
        fig.canvas.draw()
    amp_slider = Slider(ax=axbin, label="Bins", valmin=200, valmax=1500, valinit=nbin, orientation="vertical",
                        valstep=1.)
    amp_slider.on_changed(update)
    print('keV/ADU for heat: ' + str(z[0]) + ' & error ' + str(np.sqrt(np.diag(cov))[0]))
    print('offset: ' + str(z[1]))
    x1, x2 = axRAW.get_xlim()
    axcalib.set_xlim(p(x1), p(x2))

    axcalib.set_xlabel('Energy in keV')
    axRAW.set_xlabel('Amplitude in arbitrary unit')
    axRAW.set_ylabel('Number of event/'+str(np.round((p(x2)-p(x1))/nbin,1))+' keV')
    plt.show()
