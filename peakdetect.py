import numpy as np
from itertools import combinations
from scipy.stats import linregress
from scipy.signal import find_peaks, peak_prominences
from matplotlib.widgets import Slider
import get_data
import matplotlib.pyplot as plt
from cuts import cut_function
from matplotlib.widgets import Button


def find_closest_element(arr, target):
    index = np.abs(arr - target).argmin()
    return index
def get_spectrum_peaks(spectrum, threshold):
    peaks, pro = find_peaks(spectrum, prominence=threshold)
    return peaks,pro

def find_best_linear_fit(larger_list, smaller_list, threshold_percentage=10,peak_safe = 0.2, energy_safe = 4780):
    larger_list.sort()
    best_fit = None
    best_rss = float('inf')
    best_params = None
    threshold_rss = None
    threshold_params = []
    index_safe = find_closest_element(larger_list,peak_safe)
    peak_safe_real = larger_list[index_safe]
    larger_list_extracted=np.delete(larger_list,index_safe)

    for combination in combinations(larger_list_extracted, len(smaller_list)):
        selected_elements = np.append(combination,peak_safe_real)
        smaller_array = np.append(smaller_list,energy_safe)

        slope, intercept, r_value, p_value, std_err = linregress(selected_elements, smaller_array)
        rss = np.sum((slope * selected_elements + intercept - smaller_array) ** 2)

        if slope < 0:
            continue

        if rss < best_rss:
            best_rss = rss
            best_fit = selected_elements
            best_params = (slope, intercept)

        if threshold_rss is None or rss <= threshold_rss:
            threshold_rss = rss
            threshold_params.append((slope, intercept, rss))


    threshold_value = threshold_percentage * best_rss / 100.0
    threshold_params = [params for params in threshold_params if np.abs(best_rss - params[2]) <= threshold_value]

    return best_fit, best_params, threshold_params


def plot_histo(data, energies, best_params=None, best_fit=None, energy_safe = 4800., peak_safe=0.2):
    if best_fit is None:
        best_fit = np.zeros_like(energies)
    if best_params is None:
        best_params = [1, 0]
    figlin, axlin = plt.subplots()
    function_fit = np.poly1d(best_params)
    linfit, = axlin.plot(best_fit, function_fit(best_fit))
    linscat, = axlin.plot(best_fit, energies,marker='o', linestyle='')
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    n_bins = 400
    hist, bins = np.histogram(data, bins=n_bins)
    center = (bins[:-1] + bins[1:]) / 2
    hist_plot, = ax.plot(center, hist, linewidth=.5, ds='steps-mid')

    # Create slider
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Bins', valmin=1, valmax=max_bin, valinit=n_bins, valstep=1)
    axcalib = ax.twiny()

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
    print(best_fit,center)
    index_peak = [np.argmin(np.abs(center - peak)) for peak in best_fit]
    print(index_peak)
    scat_t, = ax.plot(best_fit, hist[index_peak],marker='o', linestyle='')
    scat,= ax.plot(best_fit, hist[index_peak],marker='o', linestyle='')
    def extract_peaks(_):
        nonlocal best_params
        nonlocal energies
        nonlocal best_fit

        n = int(slider.val)
        hist, bins = np.histogram(data, bins=n)
        center = (bins[:-1] + bins[1:]) / 2
        threshold = np.mean(hist)*0.5
        peaks, _ = find_peaks(hist, distance=10)
        prominences = peak_prominences(hist, peaks)[0]
        sorted_indices = np.argsort(prominences)[::-1]
        n_peaks = 8
        index_peaks = peaks[sorted_indices[:n_peaks]]
        peaks = center[index_peaks]
        scat_t.set_xdata(peaks)
        scat_t.set_ydata(hist[index_peaks])
        best_fit, best_params, threshold_params = find_best_linear_fit(peaks,energies,100,peak_safe = peak_safe, energy_safe = energy_safe)

        arg_fit=[]
        print(best_params)
        print(best_fit)
        for k in best_fit:
            arg_fit.append(np.argwhere(center == k))
        arg_fit=np.array(arg_fit)
        scat.set_xdata(best_fit)
        scat.set_ydata(hist[arg_fit])
        p = np.poly1d(best_params)
        x1, x2 = ax.get_xlim()
        axcalib.set_xlim(p(x1), p(x2))
        lin_plot(axlin,linfit, linscat, best_fit, np.append(energies,energy_safe), p)
        fig.canvas.draw_idle()
        figlin.canvas.draw_idle()
    def lin_plot(ax,linfit,linscat,xdata,ydata,f):
        linfit.set_xdata(xdata)
        linfit.set_ydata(f(xdata))
        linscat.set_xdata(xdata)
        linscat.set_ydata(ydata)
        ax.relim()
        ax.autoscale_view()


    extract_button_ax = plt.axes([0.77, 0.75, 0.1, 0.1])  # Define button axes coordinates
    extract_button = Button(extract_button_ax, 'Extract Peaks')
    extract_button.on_clicked(extract_peaks)


    def save_calib(_):
        nonlocal best_params
        nonlocal best_fit
        nonlocal energies
        get_data.update_dict(path + "dictionary.json", "heat_calib", best_params)
        get_data.update_dict(path + "dictionary.json", 'calibration_peaks', best_fit)
        get_data.update_dict(path + "dictionary.json", 'calibration_energies', np.append(energies, energy_safe))

        print('parameters saved')
    save_button_ax = plt.axes([0.65, 0.75, 0.1, 0.1])  # Define button axes coordinates
    save_button = Button(save_button_ax, 'Save')
    save_button.on_clicked(save_calib)


    slider.on_changed(update)

    plt.show()
if __name__=="__main__":
    from stabilize import get_amp_stabilize

    path, dictionary = get_data.get_path()
    filename = dictionary['filename']
    peaks = get_data.ntd_array(path + filename)
    amplitude = get_amp_stabilize(dictionary, peaks)
    cut_test = np.ones_like(amplitude, dtype=bool)
    correlation = peaks[cut_test, 5]
    amplitude = amplitude[cut_test]
    try:
        para_correlation = dictionary['correlation']
        print('cut from dictionary')
        selection = np.logical_and(correlation > cut_function(amplitude, *para_correlation), amplitude < 0.25)
    except KeyError:
        print('no correlation cut found')
        selection = amplitude < 0.25
    data = amplitude[selection]
    max_bin = 1500
    try:
        best_fit = dictionary['calibration_peaks']
        para_heat = dictionary["heat_calib"]
        energies = dictionary["calibration_energies"]
        plot_histo(data, energies[:-1],best_params=para_heat,best_fit=best_fit[:-1],energy_safe=energies[-1],peak_safe=best_fit[-1])
    except KeyError:
        energies = np.array([351,609.312, 1120, 1764])
        plot_histo(data, energies,peak_safe=0.1)