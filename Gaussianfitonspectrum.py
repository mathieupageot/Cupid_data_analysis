import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import dictionary_handler


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_lin(x, a, x0, sigma, b, c):
    return gaussian(x, a, x0, sigma) + b*(x-x0) + c


def lin_2d(x, a, b, c):
    return a*x**2+b*x+c


def inv_second_order(x,a,b,c):
    d = b/(2*a)
    return -np.sqrt((x-c)/a+d**2)-d


def positive_int(value):
    if value >= 0:
        return int(value)
    else:
        return 0


def gaussian_fit_peaks(hist, center, peaks_pos, sigma_guess=10.):
    bin_value = center[1]-center[0]
    n_wild, n_background = int(10*sigma_guess/bin_value), int(5*sigma_guess/bin_value)
    popts = []
    pcovs = []
    for peak_index in peaks_pos:
        coord_min, coord_max = np.abs(peak_index-n_wild), np.min([peak_index+n_wild,len(center)-1])
        #print(coord_min, coord_max)
        x_data = center[coord_min: coord_max]
        y_data = hist[coord_min: coord_max]
        a0 = hist[peak_index]
        x0 = center[peak_index]
        sigma0 = (x0 - x_data[0])*0.5
        mean_left = np.mean(hist[peak_index-n_wild-n_background:peak_index-n_wild])
        mean_right = np.mean(hist[peak_index+n_wild:peak_index+n_wild+n_background])
        b0 = (mean_right - mean_left) / (center[peak_index+n_wild+n_background//2] -
                                         center[peak_index-n_wild-n_background//2])
        c0 = (mean_left+mean_right)/2
        cmin, cmax = np.sort([c0*0.5,c0*2])
        if b0 != 0:
            bmin, bmax = np.sort([b0*0.5,b0*2])
        else :
            bmin, bmax = 0, 1e-3
        bounds = ([0, -np.inf, 0, bmin, cmin], [np.inf, np.inf, np.inf, bmax, cmax])
        try :
            popt, pcov = curve_fit(gauss_lin, x_data, y_data, [a0, x0, sigma0, b0, c0],
                                   bounds=bounds)
        except RuntimeError :
            popt, pcov = [a0, x0, sigma0, b0, c0], np.ones((5,5))
            print('pb with the fit of the peak at '+ str(x0))
        except ValueError:
            popt, pcov = [a0, x0, sigma0, 0, 0], np.ones((5, 5))
        popts.append(popt)
        pcovs.append(pcov)
    return np.array(popts), np.array(pcovs)


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


def update_fit(peaks_pos, popts, fits, texts):
    x = np.linspace(-1, 1, 100)
    for peak_index, popt, fit, text in zip(peaks_pos, popts, fits,texts):
        a, x0, sigma, b, c = popt
        xplot = x * sigma * 5 + x0
        fit.set_xdata(xplot)
        fit.set_ydata(gauss_lin(xplot, a, x0, sigma, b, c))
        text.set_text("x0 = {:.2e}\n$\sigma$={:.2e}".format(x0,sigma))
        text.set_y(gauss_lin(x0, a, x0, sigma, b, c)*1.5)
    return fits


def update_calib(popts, calibration_energies, plot_calib, scat_calib, dictionary, x_calib):
    p_calib, cov_calib = curve_fit(lin_2d, calibration_energies, popts[:, 1],
                                   p0=[0, 1 / dictionary['heat_calib'][0], 0], sigma=popts[:, 2])
    plot_calib.set_ydata(lin_2d(x_calib, *p_calib))
    scat_calib.set_ydata(popts[:, 1])

def plot_hist_bin(data,dictionary, n0=100,sigma_guess=30., calibrated=True):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom = 0.25)
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    counts, bins = np.histogram(data, n0)
    center = (bins[:-1] + bins[1:]) / 2
    hist, = ax.plot(center,counts,ds='steps-mid')
    slider_bin = Slider(ax_slider, 'bins', valmin=n0*0.1, valmax=n0*10, valinit=n0, valstep=1)
    if calibrated :
        calib_function = np.poly1d(dictionary['heat_calib'])
        values = calib_function(dictionary['calibration_peaks'])
        ax.set_ylabel("Number of event/{:.2e} keV".format(center[1] - center[0]))
        ax.set_xlabel("Energy in keV")
    else:
        values = dictionary['calibration_peaks']
        ax.set_ylabel("Number of event/{:.2e} ADU".format(center[1] - center[0]))
        ax.set_xlabel("Energy in ADU")
    peaks_pos = []
    for value in values:
        peaks_pos.append(np.abs(center - value).argmin())
    popts, _ = gaussian_fit_peaks(counts, center, peaks_pos,sigma_guess=sigma_guess)
    if not calibrated:
        calibration_energies = dictionary['calibration_energies']
        p_calib, cov_calib = curve_fit(lin_2d, calibration_energies, popts[:,1],
                               p0=[0,1/dictionary['heat_calib'][0],0],sigma=popts[:,2])
        fig_calib, ax_calib = plt.subplots()
        x_calib = np.linspace(0,6000.,1000)
        plot_calib, = ax_calib.plot(x_calib, lin_2d(x_calib,*p_calib))
        scat_calib, = ax_calib.plot(calibration_energies, popts[:,1], marker='o', linestyle='')
    fits,texts = plot_fit(peaks_pos, popts, ax)
    def update_bin(_):
        nonlocal p_calib, cov_calib
        nbins = int(slider_bin.val)
        upd_counts, upd_bins = np.histogram(data, nbins)
        upd_center = (upd_bins[:-1] + upd_bins[1:]) / 2
        ax.set_ylabel("Number of event/{:.2e} keV".format(upd_center[1] - upd_center[0]))
        hist.set_ydata(upd_counts)
        hist.set_xdata(upd_center)
        upd_peaks_pos = []
        for value in values:
            upd_peaks_pos.append(np.abs(upd_center - value).argmin())
        upd_popts, pcovs = gaussian_fit_peaks(upd_counts, upd_center, upd_peaks_pos, sigma_guess=sigma_guess)
        update_fit(upd_peaks_pos, upd_popts, fits, texts)
        ax.set_ylim(-10,upd_counts.max())
        if not calibrated:
            update_calib(upd_popts, calibration_energies, plot_calib, scat_calib, dictionary, x_calib)
            fig_calib.canvas.draw_idle()
        fig.canvas.draw_idle()
    slider_bin.on_changed(update_bin)
    def save_para(_):
        nonlocal p_calib, cov_calib
        dictionary_handler.update_dict(dictionary['path']+"dictionary.json",
                                       {"heat_calib_opti": p_calib.tolist(),
                                        'heat_calib_opti_error': np.sqrt(np.diag(cov_calib)).tolist()})
        print("heat_calib_opti saved")


    save_button_ax = fig_calib.add_axes([0.77, 0.15, 0.1, 0.1])  # Define button axes coordinates
    save_button = Button(save_button_ax, 'Save')
    save_button.on_clicked(save_para)
    plt.show()
if __name__ == "__main__":
    import get_data
    from cuts import cut_function
    path, dictionary = get_data.get_path()
    if 'filename' in dictionary:
        E, correlation, amplitude = get_data.get_pulses(dictionary,['Energy','Correlation','Amplitude_filtered'])
        try:
            para_correlation = dictionary['correlation']
            selection = np.logical_and(correlation > cut_function(amplitude, *para_correlation), E<6000)
        except KeyError:
            print('no correlation cut found')
            selection = np.ones_like(E,dtype=bool)
        E_sel = E[selection]
        plot_hist_bin(amplitude[amplitude<1.3], dictionary, n0=3000, sigma_guess=0.002, calibrated=False)

