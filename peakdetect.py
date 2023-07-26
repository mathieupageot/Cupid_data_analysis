import numpy as np
from itertools import combinations
from scipy.stats import linregress
from scipy.signal import find_peaks, peak_prominences
from matplotlib.widgets import Slider
from utility.double_axis_display import make_format
import dictionary_handler
import get_data
import matplotlib.pyplot as plt
from cuts import cut_function
from matplotlib.widgets import Button
from scipy.optimize import curve_fit


def linear(x,a,b):
    return a*x+b


def find_closest_element(arr, targets):
    index = []
    for target in targets:
        index.append(np.abs(arr - target).argmin())
    return index


def get_spectrum_peaks(spectrum, threshold):
    peaks, pro = find_peaks(spectrum, prominence=threshold)
    return peaks, pro


def find_best_linear_fit(larger_list, smaller_list, threshold_baseline,
                         threshold_percentage, peak_safe, energy_safe, use_peaksafe = True):
    p_guess = [energy_safe[-1] / peak_safe[-1], 0]
    print(p_guess)
    bounds = ([0, -threshold_baseline*p_guess[0]], [np.inf, threshold_baseline*p_guess[0]])
    larger_list.sort()
    best_fit = None
    best_rss = float('inf')
    best_params = None
    threshold_rss = None
    best_cov = None
    threshold_params = []
    index_safe = find_closest_element(larger_list, peak_safe)
    peak_safe_real = larger_list[index_safe]
    larger_list_extracted = np.delete(larger_list, index_safe)

    for combination in combinations(larger_list_extracted, len(smaller_list)):
        if use_peaksafe:
            selected_elements = np.append(combination, peak_safe_real)
            smaller_array = np.append(smaller_list, energy_safe)
        else :
            selected_elements = np.array(combination)
            smaller_array = np.array(smaller_list)
        popt, pcov = curve_fit(linear, selected_elements, smaller_array, p0=p_guess, bounds=bounds)
        slope, intercept = popt
        rss = np.sum((slope * selected_elements + intercept - smaller_array) ** 2)

        if slope < 0:
            continue

        if rss < best_rss:
            best_rss = rss
            best_fit = selected_elements
            best_params = (slope, intercept)
            best_cov = pcov

        if threshold_rss is None or rss <= threshold_rss:
            threshold_rss = rss
            threshold_params.append((slope, intercept, rss))

    threshold_value = threshold_percentage * best_rss / 100.0
    threshold_params = [params for params in threshold_params if np.abs(best_rss - params[2]) <= threshold_value]

    return best_fit, best_params, threshold_params, best_cov


def get_calibration(data, energies, dictionary, best_params=None, best_fit=None, energy_safe=4800., peak_safe=0.2):
    path = dictionary['path']
    energy_safe = np.array(energy_safe,dtype=float).reshape(-1)
    peak_safe = np.array(peak_safe,dtype=float).reshape(-1)
    if best_fit is None:
        best_fit = np.zeros_like(energies)
    if best_params is None:
        best_params = [1, 0]
    best_cov = np.zeros((2,2))
    n_peaks = (len(energies)+len(energy_safe))*2
    figlin, axlin = plt.subplots()
    axlin.set_xlabel('Amplitude in ADU')
    axlin.set_ylabel('Energy in keV')
    function_fit = np.poly1d(best_params)
    linfit, = axlin.plot(np.append(best_fit, peak_safe), function_fit(np.append(best_fit, peak_safe)))
    linscat, = axlin.plot(np.append(best_fit, peak_safe), np.append(energies, energy_safe), marker='o', linestyle='')
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    n_bins = 585
    hist, bins = np.histogram(data, bins=n_bins)
    center = (bins[:-1] + bins[1:]) / 2
    hist_plot, = ax.plot(center, hist, linewidth=.5, ds='steps-mid')

    # Create slider
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Bins', valmin=1, valmax=max_bin, valinit=n_bins, valstep=1)
    axcalib = ax.twiny()
    axcalib.format_coord = make_format(axcalib, ax)
    p = np.poly1d(best_params)
    x1, x2 = ax.get_xlim()
    axcalib.set_xlim(p(x1), p(x2))
    ax.set_yscale('log')
    ax.set_ylabel("Counts / {:.2e} keV".format(p(bins[1]-bins[0])))
    ax.set_xlabel('Amplitude in ADU')
    axcalib.set_xlabel('Energy in keV')

    # Function to update histogram based on slider value
    def update(val):
        n = int(slider.val)
        hist, bins = np.histogram(data, bins=n)
        center = (bins[:-1] + bins[1:]) / 2
        hist_plot.set_xdata(center)
        hist_plot.set_ydata(hist)
        ax.set_ylim(0, hist.max())

        ax.set_xlim(bins[0], bins[-1])
        fig.canvas.draw_idle()
    index_peak = [np.argmin(np.abs(center - peak)) for peak in np.append(best_fit, peak_safe)]
    scat_t, = ax.plot(np.append(best_fit, peak_safe), hist[index_peak], marker='o', linestyle='')
    scat, = ax.plot(np.append(best_fit, peak_safe), hist[index_peak], marker='o', linestyle='')

    def extract_peaks(_):
        nonlocal best_params
        nonlocal energies
        nonlocal best_fit
        nonlocal n_peaks
        nonlocal best_cov
        n = int(slider.val)
        hist, bins = np.histogram(data, bins=n)
        center = (bins[:-1] + bins[1:]) / 2
        threshold = np.mean(hist) * 0.5
        peaks, _ = find_peaks(hist, distance=10)
        prominences = peak_prominences(hist, peaks)[0]
        sorted_indices = np.argsort(prominences)[::-1]
        index_peaks = peaks[sorted_indices[:n_peaks]]
        peaks = center[index_peaks]
        scat_t.set_xdata(peaks)
        scat_t.set_ydata(hist[index_peaks])
        best_fit, best_params, threshold_params, best_cov = find_best_linear_fit(peaks, energies,
                                                                                 dictionary['threshold'], 100,
                                                                       peak_safe=peak_safe, energy_safe=energy_safe,
                                                                       use_peaksafe=True)

        arg_fit = []
        print(best_params)
        print(best_fit)
        for k in best_fit:
            arg_fit.append(np.argwhere(center == k))
        arg_fit = np.array(arg_fit)
        scat.set_xdata(best_fit)
        scat.set_ydata(hist[arg_fit])
        p = np.poly1d(best_params)
        x1, x2 = ax.get_xlim()
        axcalib.set_xlim(p(x1), p(x2))
        lin_plot(axlin, linfit, linscat, best_fit, np.append(energies, energy_safe), p)
        fig.canvas.draw_idle()
        figlin.canvas.draw_idle()

    def lin_plot(ax, linfit, linscat, xdata, ydata, f):
        linfit.set_xdata(xdata)
        linfit.set_ydata(f(xdata))
        linscat.set_xdata(xdata)
        linscat.set_ydata(ydata)

    extract_button_ax = plt.axes([0.77, 0.75, 0.1, 0.1])  # Define button axes coordinates
    extract_button = Button(extract_button_ax, 'Extract Peaks')
    extract_button.on_clicked(extract_peaks)

    def save_calib(_):
        nonlocal best_params
        nonlocal best_fit
        nonlocal energies
        nonlocal energy_safe
        nonlocal best_cov
        nonlocal path
        dictio_update = {"heat_calib": best_params, 'calibration_peaks': list(best_fit),
         'calibration_energies': list(np.concatenate((energies, energy_safe))),
                         'heat_calib_error':list(np.sqrt(np.diag(best_cov)))}
        dictionary_handler.update_dict(path + "dictionary.json",
                                       dictio_update)
        print('parameters saved in' + path + "dictionary.json")

    save_button_ax = plt.axes([0.65, 0.75, 0.1, 0.1])  # Define button axes coordinates
    save_button = Button(save_button_ax, 'Save')
    save_button.on_clicked(save_calib)
    def safe_calib(_):
        nonlocal best_fit
        nonlocal energy_safe
        print(best_fit)
        dictio_update = {"heat_calib": [energy_safe[-1]/best_fit[-1],0], 'calibration_peaks': [0,best_fit[-1]],
                         'calibration_energies': [0, energy_safe[-1]] }
        dictionary_handler.update_dict(path + "dictionary.json",
                                       dictio_update)
        print('parameters saved in' + path + "dictionary.json")
    safe_button_ax = plt.axes([0.54, 0.75, 0.1, 0.1])  # Define button axes coordinates
    safe_button = Button(safe_button_ax, 'Safe calib')
    safe_button.on_clicked(safe_calib)
    slider.on_changed(update)

    plt.show()


def compute_fit_error(path,peaks,energies, para):
    popt, pcov = curve_fit(linear,peaks,energies,para)
    dictionary_handler.update_dict(path + 'dictionary.json', {'heat_calib_error': np.sqrt(np.diag(pcov)).tolist()})

if __name__ == "__main__":
    '''
    import matplotlib.pylab as pylab
    save_plot = 1
    if save_plot == 1:
        import matplotlib.pylab as pylab

        params = {'legend.fontsize': 20.,
                  'figure.figsize': (15, 5),
                  'axes.labelsize': 20.,
                  'axes.titlesize': 20.,
                  'xtick.labelsize': 20.,
                  'ytick.labelsize': 20.}
        pylab.rcParams.update(params)
    path, dictionary = get_data.get_path()
    amplitude, correlation = get_data.get_pulses(dictionary,['amp_stab', 'Correlation'])
    try:
        para_correlation = dictionary['correlation']
        print('cut from dictionary')
        selection = correlation > cut_function(amplitude, *para_correlation)
    except KeyError:
        print('no correlation cut found')
        selection = amplitude < 0.2
    data = amplitude[selection]
    max_bin = 1500
    change_peak_safe = False
    peak_safe = np.array([0.0063])
    n_safe = len(peak_safe)
    try:
        best_fit = dictionary['calibration_peaks']
        para_heat = dictionary["heat_calib"]
        if not change_peak_safe:
            peak_safe = best_fit[-1]
            n_safe = 1
        energies = dictionary["calibration_energies"]
        data = data[data<peak_safe*1.1]
        get_calibration(data, energies[:-1], dictionary, best_params=para_heat, best_fit=best_fit[:-n_safe],
                        energy_safe=energies[-n_safe], peak_safe=peak_safe)
    except KeyError:
        data = data[data < peak_safe * 1.1]
        energies = np.array([352,609,1120,1764])
        #energies = np.array([511, 1461])
        get_calibration(data, energies, dictionary, peak_safe=peak_safe, energy_safe=[5000])
'''
    for j in [3, 4, 5, 6, 7, 8, 11, 12]:
        path, dictionary = get_data.get_path(11,j)
        peaks, energies, para = dictionary_handler.get_values(dictionary,['calibration_peaks',
                                                                          'calibration_energies', 'heat_calib'])
        compute_fit_error(path, peaks, energies, para)


