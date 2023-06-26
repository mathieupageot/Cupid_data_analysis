import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_lin(x, a, x0, sigma, b, c):
    return gaussian(x, a, x0, sigma) + b*(x-x0) + c


def gaussian_fit_peaks(hist,center,peaks_pos):
    n_wild = int(len(hist)*0.01)
    popts = []
    pcovs = []
    for peak_index in peaks_pos:
        x_data = center[peak_index-n_wild:peak_index+n_wild]
        y_data = hist[peak_index-n_wild:peak_index+n_wild]
        a0 = hist[peak_index]
        x0 = center[peak_index]
        sigma0 = (x0 - x_data[0])*0.5
        b0 = (y_data[0] - y_data[-1]) / (x_data[0] - x_data[-1])
        c0 = 0
        try :
            popt, pcov = curve_fit(gauss_lin, x_data, y_data, [a0, x0, sigma0, b0, c0])
        except RuntimeError:
            popt = [a0, x0, sigma0, b0, c0]
            print('pb with the fit of the peak at '+ str(x0))
        popts.append(popt)
        pcovs.append(pcov)
    return popts, pcovs


def get_peaks(x_data, y_data, prominence_threshold, n_peaks):
    n_comp = len(y_data[x_data <= 500])
    peaks_pos, peaks_param = find_peaks(y_data[x_data>500], prominence=prominence_threshold)
    peaks_prominences = peaks_param['prominences']
    return peaks_pos[np.argpartition(peaks_prominences, -n_peaks)[-n_peaks:]] + n_comp


def plot_fit(peaks_pos, popts, ax):
    fits = []
    texts = []
    x = np.linspace(-1, 1, 100)
    for peak_index, popt in zip(peaks_pos, popts) :
        a, x0, sigma, b, c = popt
        xplot = x*sigma*5+x0
        fit, = ax.plot(xplot, gauss_lin(xplot, a, x0, sigma, b, c))
        text = ax.text(x0, gauss_lin(x0, a, x0, sigma, b, c)*1.5,"x0 = {:.2e}\n$\sigma$={:.2e}".format(x0,sigma), fontsize=12, color='red')
        fits.append(fit)
        texts.append(text)
    return fits, texts


def update_fit(peaks_pos, popts, fits, texts, pcovs):
    x = np.linspace(-1, 1, 100)
    print(repr(np.array(popts)[:,1:3]))
    print(repr(np.sqrt(np.diagonal(np.array(pcovs),axis1=1,axis2=2))[:,1:3]))
    for peak_index, popt, fit, text in zip(peaks_pos, popts, fits,texts):
        a, x0, sigma, b, c = popt
        xplot = x * sigma * 5 + x0
        fit.set_xdata(xplot)
        fit.set_ydata(gauss_lin(xplot, a, x0, sigma, b, c))
        text.set_text("x0 = {:.2e}\n$\sigma$={:.2e}".format(x0,sigma))
        text.set_y(gauss_lin(x0, a, x0, sigma, b, c)*1.5)
    return fits


def plot_hist_bin(data,n0=100):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom = 0.25)
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    counts, bins = np.histogram(data, n0)
    center = (bins[:-1] + bins[1:]) / 2
    hist, = ax.plot(center,counts,ds='steps-mid')
    slider_bin = Slider(ax_slider, 'bins', valmin=n0*0.1, valmax=n0*10, valinit=n0, valstep=1)
    values = [583.191,968.971,1473, 2614.533]
    peaks_pos = []
    for value in values:
        peaks_pos.append(np.abs(center - value).argmin())

    popts, _ = gaussian_fit_peaks(counts, center, peaks_pos)
    ax.set_ylabel("Number of event/{:.2e} keV".format(center[1]-center[0]))
    ax.set_xlabel("Energy in keV")

    fits,texts = plot_fit(peaks_pos, popts, ax)
    def update_bin(_):
        nbins = int(slider_bin.val)
        upd_counts, upd_bins = np.histogram(data, nbins)
        upd_center = (upd_bins[:-1] + upd_bins[1:]) / 2
        ax.set_ylabel("Number of event/{:.2e} keV".format(upd_center[1] - upd_center[0]))
        hist.set_ydata(upd_counts)
        hist.set_xdata(upd_center)
        upd_peaks_pos = []
        for value in values:
            upd_peaks_pos.append(np.abs(upd_center - value).argmin())
        upd_popts, pcovs = gaussian_fit_peaks(upd_counts, upd_center, upd_peaks_pos)
        update_fit(upd_peaks_pos, upd_popts, fits, texts, pcovs)
        ax.set_ylim(-10,upd_counts.max())
        fig.canvas.draw_idle()
    slider_bin.on_changed(update_bin)
    plt.show()
if __name__ == "__main__":
    import get_data
    from cuts import cut_function
    path, filename, filename_light, filename_trigheat, p = get_data.get_path()
    if filename != 0:
        E, amp, correlation, TV, *_ = get_data.get_heat(path, filename, p)
        try:
            para_corr = np.load(path+filename.strip(".ntp")+'_'+'correlation'+".npy")
        except FileNotFoundError:
            print('No correlation cut found')
            para_corr = np.array([-1,-1,0.80])
        correl_cut = correlation > 0
        cut = np.logical_and(correl_cut, E < 3000)
        E = E[cut]
        plot_hist_bin(E, n0=3000)