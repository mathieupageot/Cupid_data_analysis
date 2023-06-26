import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from landaupy import langauss
from scipy.stats import median_abs_deviation
from matplotlib.widgets import Slider, RectangleSelector, Button


def gaussian(x, x0, a, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def get_energy_mu(filename_light):
    if filename_light == '20180709_23h07.BINLD.ntp':
        energy_mu = 100
    elif filename_light == '20211125_00h43.BINLD21.2.ntp':
        energy_mu = 260
    elif filename_light[7:15] == '20230609':
        energy_mu = 260
    elif filename_light[7:13] == '202306':
        energy_mu = 210
    else:
        energy_mu = 100
    print('energy of the muon : ',energy_mu)
    return energy_mu


def binned_fit_langauss(hist, bin_centers, samples):
    landau_x_mpv_guess = bin_centers[np.argmax(hist)]
    landau_xi_guess = median_abs_deviation(samples) / 5
    gauss_sigma_guess = landau_xi_guess / 10
    x0_guess = bin_centers[0]
    y0_guess = hist[0]
    print(x0_guess,landau_x_mpv_guess,landau_xi_guess,gauss_sigma_guess)
    popt, pcov = curve_fit(
        lambda x, mpv, xi, sigma, x0, y0: langauss.pdf(x-x0, mpv, xi, sigma)+y0,
        xdata=bin_centers,
        ydata=hist,
        p0=[landau_x_mpv_guess, landau_xi_guess, gauss_sigma_guess,x0_guess,y0_guess],
    )
    return popt, pcov


def binned_fit_gauss(hist, bin_centers, samples):
    a_guess = np.max(hist)
    x0_guess = bin_centers[np.argmax(hist)]
    sigma_guess = np.std(samples)
    popt, pcov = curve_fit(gaussian,bin_centers,hist,[x0_guess,a_guess,sigma_guess])
    return popt, pcov


def hist_light_fit(ampl, filename_light, path):
    fig4, axs4 = plt.subplots(1)
    plt.subplots_adjust(bottom=0.20)
    nbins = 200
    hist, bin_edges = np.histogram(ampl, nbins, density=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    plot_histogram, = axs4.plot(bin_centers, hist, linewidth=.5, ds='steps-mid')
    x_plot = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    plot_fit, = axs4.plot(x_plot, np.zeros_like(x_plot))
    text_fit = axs4.text(0.6, 0.60, "Parameters of the fit :\n$x_{{MPV}}$ :NaN}\nξ : NaN\n$\sigma$ : NaN",
                         fontsize=12, color='red', transform=axs4.transAxes)
    axs4.text(0.1, 0.95, "Press e key to save the current parameters",
              fontsize=12, color='red', transform=axs4.transAxes)
    popt, pcov = 0, 0
    axs4.set_ylim(0, hist.max())
    axbins = plt.axes([0.1, 0.05, 0.65, 0.03])
    slider_bins = Slider(axbins, 'bins', 0.1*nbins, 10*nbins, valinit=nbins)

    def rebins(event):
        nonlocal nbins
        nonlocal bin_centers
        nonlocal hist
        nbins = slider_bins.val
        hist, bin_edges = np.histogram(ampl, nbins, density=True)
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        plot_histogram.set_xdata(bin_centers)
        plot_histogram.set_ydata(hist)
        fig4.canvas.draw_idle()


    def line_select_callback(eclick, erelease):
        nonlocal popt
        nonlocal pcov
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        toggle_selector.RS.set_active(True)
        bool_hist_x = np.logical_and(bin_centers < np.max((x1, x2)), bin_centers > np.min((x1, x2)))
        bool_hist_y = np.logical_and(hist < np.max((y1, y2)), hist > np.min((y1, y2)))
        bool_hist = np.logical_and(bool_hist_x, bool_hist_y)
        bool_ampl = np.logical_and(ampl < np.max((x1, x2)), ampl > np.min((x1, x2)))
        try:
            # popt, pcov = binned_fit_langauss(hist[bool_hist], bin_centers[bool_hist], ampl[bool_ampl])
            # text_fit.set_text(
            #     "Parameter of the fit :\n$x_{{MPV}}$ : {:.2e}\nξ : {:.2e}\n$\sigma$ : {:.2e}".format(*popt))
            # plot_fit.set_xdata(x_plot)
            # plot_fit.set_ydata(langauss.pdf(x_plot-popt[-2], *popt[:-2])+popt[-1])
            popt, pcov = binned_fit_gauss(hist[bool_hist], bin_centers[bool_hist], ampl[bool_ampl])
            text_fit.set_text(
                "Parameter of the fit :\nx0 : {:.2e}\nA : {:.2e}\n$\sigma$ : {:.2e}".format(*popt))
            plot_fit.set_xdata(x_plot)
            plot_fit.set_ydata(gaussian(x_plot,*popt))

        except TimeoutError:
            text_fit.set_text("Parameter of the fit :\n$x_{{MPV}}$ :NaN}\nξ : NaN\n$\sigma$ : NaN")

        fig4.canvas.draw_idle()

    def toggle_selector(event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
        if event.key in ['E', 'e']:
            energy_mu = get_energy_mu(filename_light)
            get_data.update_dict(path + "dictionary.json", "light_calib",
                                 [energy_mu / popt[0], np.sqrt(pcov[0, 0]) * energy_mu / popt[0] ** 2])
            print("light calibration saved")

    toggle_selector.RS = RectangleSelector(axs4, line_select_callback,
                                           useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    slider_bins.on_changed(rebins)
    axs4.set_xlabel("Light amplitude in arbitrary unit")
    axs4.set_ylabel("Counts/{:.0e} arbitrary units".format((bin_centers[-1] - bin_centers[0]) / len(bin_centers)))
    axs4.set_title('Landeau fit of the light channel')


if __name__ == '__main__':
    import get_data

    path, dictionary = get_data.get_path()
    filename_light = dictionary["filename_light"]
    if filename_light != 0:
        amplitude_light,riset = get_data.import_light(path, filename_light)
        hist_light_fit(amplitude_light, filename_light, path)
        A=dictionary['light_calib']
    plt.show()
