import numpy as np
import matplotlib.pyplot as plt
import get_data
from cuts import cut_function
import risetime
from matplotlib.widgets import Slider, RectangleSelector
from scipy.optimize import curve_fit

def change_frontsize(axes,frontsize):
    for ax in axes:
        ax.tick_params(axis="x", labelsize=frontsize)
        ax.tick_params(axis="y", labelsize=frontsize)


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def double_gaussian(x, amp1, center1, sigma1, amp2, center2, sigma2):
    gaussian1 = amp1 * np.exp(-(x - center1)**2 / (2 * sigma1**2))
    gaussian2 = amp2 * np.exp(-(x - center2)**2 / (2 * sigma2**2))
    return gaussian1 + gaussian2
gaussian_dict = {'single' : gaussian, 'double' : double_gaussian}

def f_tv(x, a, b, c):
    return np.exp((x - a) / b) * c
def picker(ax):
    fig = ax.get_figure()
    def onpick(event):
        ind = event.ind
        print('onpick scatter:', time[ind]-3000)
    fig.canvas.mpl_connect('pick_event', onpick)
def plot_TV(E,TV,TV_cut,para_TV,ax):
    ax.scatter(E, TV, s=0.1)
    ax.scatter(E[TV_cut], TV[TV_cut], s=0.1)
    xp = np.linspace(0, 80000)
    ax.plot(xp, cut_function(xp, *para_TV))
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylim(0,TV.max())
    ax.set_ylabel('TV')
    ax.set_title('Energy vs TV')


def plot_rise(E, riset, ax):
    risetime.plot_scatter_with_gauss(E[E<20],riset[E<20],ax,len(E)//30)
    ax.set_ylim(-0.02,0.06)
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylabel('Rise time in s')
    ax.set_title('Energy vs Rise Time')


def plot_decay(E,decayt,ax):
    ax.scatter(E, decayt, s=0.1)
    ax.set_ylim(0,0.1)
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylabel('Decay time in s')
    ax.set_title('Energy vs Decay Time')
def plot_corr(E,correlation,para,ax):
    ax.scatter(E, correlation, s=0.1)
    ax.set_ylim(0.98, 1.)
    xcor = np.linspace(E.min(), E.max(), 5000)
    ax.plot(xcor, cut_function(xcor, *para), c='r', label='correlation cut')
    ax.legend()
    ax.set_title('Correlation vs Pulse energy raw data heat channel')
    ax.set_ylabel('Correlation normalized')
    ax.set_xlabel('Pulse energy in arbitrary unit')


def hist_heat_cut(E, correl_cut, ax):
    amp_corr = E[correl_cut]
    ncorr, binscorr = np.histogram(amp_corr, 3000)
    centercorr = (binscorr[:-1] + binscorr[1:]) / 2
    ax.plot(centercorr, ncorr, linewidth=.5, ds='steps-mid', label='Correlation cut')
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylabel('Counts/2 keV')
    ax.legend()


def import_triglight(path ,filename_trigheat,sel):
    peaksl = get_data.ntd_array(path + filename_trigheat)
    ampl = peaksl[sel, 2]
    trig_rt = peaksl[sel, 11]
    return ampl, trig_rt
def plot_Sm(E,Sm,ax):
    ax.scatter(E, Sm, s=0.1)
    ax.set_xlabel('Amplitude in keV')
    ax.set_ylabel('S$_m$/m')
    ax.set_title('The Fitted Amplitude vs Energy')
    ymean=np.mean(Sm)
    yvar=np.var(Sm)
    ax.set_ylim(ymean-100*yvar,ymean+100*yvar)


def hist_light(amplt_fit,ax):
    n, bins = np.histogram(amplt_fit[amplt_fit < 6000], 500)
    center = (bins[:-1] + bins[1:]) / 2
    ax.plot(center, n, linewidth=.5, ds='steps-mid', label='Calibrated')
    ax.set_xlabel('Light Channel Energy in keV')
    ax.set_ylabel('Counts/12 keV')


def plot_light(E_sel,ampl_fit,ax):
    ax.scatter(E_sel, ampl_fit, s=0.1)
    ax.set_ylabel('Light amplitude in keV')
    ax.set_xlabel('Heat amplitude in keV')
    ax.set_title('Heat amplitude VS Light amplitude for alpha discrimination')
    ax.set_ylim(-5,5)


def plot_LY(E,LY,ax, axfit):
    fig = ax.get_figure()
    pts = ax.scatter(E,LY,s=0.1)
    ax.set_ylabel('LY ADU')
    ax.set_xlabel('Heat amplitude in keV')
    ax.set_title('Heat Energy vs LY for {}'.format(path.split("/")[-2]))
    ymean=np.mean(LY)
    yvar=np.var(LY)
    ax.set_ylim(ymean-yvar,ymean+yvar)
    ax_slider = plt.axes([0.75, 0.65, 0.2, 0.03])
    slider = Slider(ax_slider, 'bins', valmin=10, valmax=1000, valinit=100, valstep=1)
    text_parameters = ax.text(0.8, 0.53, 'No parameters yet', fontsize=12, bbox=dict(facecolor='white', edgecolor='white'),color='black', transform=ax.transAxes,verticalalignment='top')
    gaussian_type_fit = 'single'
    text_gaussian_type = ax.text(0.8, 0.60, "gaussian type : "+gaussian_type_fit,bbox=dict(facecolor='white', edgecolor='white'), fontsize=12, color='black', transform=ax.transAxes,verticalalignment='top')
    # Create a box selection event handler
    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        toggle_selector.RS.set_active(True)
        axfit.clear()
        bool_LY= np.logical_and(LY<np.max((y1,y2)),LY>np.min((y1,y2)))
        bool_E = np.logical_and(E<np.max((x1,x2)),E>np.min((x1,x2)))
        selected_LY = LY[np.logical_and(bool_LY,bool_E)]
        nbins = int(len(selected_LY)/50)+1
        slider.valmin, slider.valmax, slider.valinit = nbins*0.1, nbins*10, nbins

        n, bins = np.histogram(selected_LY, nbins)
        center = (bins[:-1] + bins[1:]) / 2
        hist, = axfit.plot(center, n, linewidth=.5, ds='steps-mid')
        datax = np.linspace(selected_LY.min(),selected_LY.max(),1000)
        if gaussian_type_fit == 'single':

            popt, pcov = curve_fit(gaussian_dict[gaussian_type_fit], center, n,
                                   [n.max(), center.mean(), np.std(center)])
            text_parameters.set_text("$LY_0$ = {:.2e} \n$\sigma$ = {:.2e}".format(popt[1], popt[2]))
        elif gaussian_type_fit == 'double':
            amp1_guess = np.max(n) - np.min(n)
            center1_guess = center[np.argmax(n)]
            sigma1_guess = (np.max(center) - np.min(center)) / 10
            amp2_guess = amp1_guess / 2
            center2_guess = center1_guess + (np.max(center) - np.min(center)) / 4
            sigma2_guess = sigma1_guess
            initial_guess = [amp1_guess, center1_guess, sigma1_guess, amp2_guess, center2_guess, sigma2_guess]
            popt, pcov = curve_fit(gaussian_dict[gaussian_type_fit], center, n,
                                   initial_guess)
            discrimination_power = np.absolute(popt[1]-popt[4])/np.sqrt(popt[2]**2+popt[5]**2)
            text_parameters.set_text("$LY_01$ = {:.2e} \n$\sigma1$ = {:.2e}\n$LY_02$ = {:.2e} \n$\sigma2$ = {:.2e}\nDiscrimination Power = {:.2e}".format(popt[1], popt[2],popt[4], popt[5],discrimination_power))
        plot, = axfit.plot(datax,gaussian_dict[gaussian_type_fit](datax, *popt))



        def update(_):
            nbins = int(slider.val)
            n, bins = np.histogram(selected_LY, nbins)
            center = (bins[:-1] + bins[1:]) / 2
            if gaussian_type_fit == 'single':
                popt, pcov = curve_fit(gaussian_dict[gaussian_type_fit],center,n,[n.max(), center.mean(), np.std(center)])
                text_parameters.set_text("$LY_0$ = {:.2e} \n$\sigma$ = {:.2e}".format(popt[1], popt[2]))
            elif gaussian_type_fit == 'double':

                amp1_guess = np.max(n) - np.min(n)
                center1_guess = center[np.argmax(n)]
                sigma1_guess = (np.max(center) - np.min(center)) / 10
                amp2_guess = amp1_guess / 2
                center2_guess = center1_guess + (np.max(center) - np.min(center)) / 4
                sigma2_guess = sigma1_guess
                initial_guess = [amp1_guess, center1_guess, sigma1_guess, amp2_guess, center2_guess, sigma2_guess]
                popt, pcov = curve_fit(gaussian_dict[gaussian_type_fit], center, n,
                                       initial_guess)
                discrimination_power = np.absolute(popt[1] - popt[4]) / np.sqrt(popt[2] ** 2 + popt[5] ** 2)
                text_parameters.set_text(
                    "$LY_01$ = {:.2e} \n$\sigma1$ = {:.2e}\n$LY_02$ = {:.2e} \n$\sigma2$ = {:.2e}\nDiscrimination Power = {:.2e}".format(
                        popt[1], popt[2], popt[4], popt[5], discrimination_power))
            plot.set_ydata(gaussian_dict[gaussian_type_fit](datax,*popt))
            hist.set_xdata(center)
            hist.set_ydata(n)
            axfit.set_ylim(0,n.max()*1.01)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        fig.canvas.draw_idle()
    def toggle_selector(event):
        print(' Key pressed.')
        nonlocal gaussian_type_fit
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
        if event.key in ['D','d']:
            gaussian_type_fit = 'double'
            text_gaussian_type.set_text("gaussian type : "+gaussian_type_fit)
        if event.key in ['O','o']:
            gaussian_type_fit = 'single'
            text_gaussian_type.set_text("gaussian type : " + gaussian_type_fit)

    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)


if __name__ == '__main__':
    import matplotlib.pylab as pylab

    params = {'legend.fontsize': 20.,
              'figure.figsize': (15, 5),
              'axes.labelsize': 20.,
              'axes.titlesize': 20.,
              'xtick.labelsize': 20.,
              'ytick.labelsize': 20.}
    pylab.rcParams.update(params)
    path, dictionary = get_data.get_path()
    axes = []
    print(dictionary)
    try :
        filename, filename_light, filename_trigheat, heatcalib = get_data.get_values(dictionary, ["filename", "filename_light", "filename_trigheat", "heat_calib"])
    except ValueError:
        print("miss the heat calib")
    heatcalib_function = np.poly1d(heatcalib)
    if filename != 0:
        E, amplitude, correlation, TV, riset, decayt, Sm, time = get_data.get_heat(path, filename, heatcalib_function, dictionary)
        try:
            para_correlation = dictionary['correlation']
            print('cut from dictionary')
            selection = np.logical_and(correlation > cut_function(amplitude, *para_correlation), amplitude < 0.25)
        except KeyError:
            print('no correlation cut found')
            selection = amplitude < 0.25
        E_sel = E[selection]
        fig_heat_histo, ax_heat_histo = plt.subplots()
        axes.append(ax_heat_histo)
        hist_heat_cut(E, selection,ax_heat_histo)
    if filename_light != 0:
        ampl,risetl = get_data.import_light(path, filename_light)
        coeff_light, error_coeff_light = dictionary["light_calib"]
        ampl_fit = ampl * coeff_light

    if filename_trigheat != 0:
        amplt, trig_rt = import_triglight(path, filename_trigheat, selection)
        amplt_fit = amplt * coeff_light
        LY = amplt_fit / E_sel
        fig_light_vs_light, ax_light_vs_light = plt.subplots()
        axes.append(ax_light_vs_light)
        plot_light(E_sel, amplt_fit, ax_light_vs_light)
        figLY, axLY = plt.subplots()
        axes.append(axLY)
        axfit = figLY.add_axes([0.75, 0.75, 0.2, 0.2])

        axes.append(axfit)
        plot_LY(E_sel, LY,axLY, axfit)
        axfit.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        fig_rise_time_light, ax_rise_time_light = plt.subplots()
        axes.append(ax_rise_time_light)
        plot_rise(amplt_fit, trig_rt, ax_rise_time_light)



    plt.show()