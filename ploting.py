import numpy as np
import matplotlib.pyplot as plt
from cuts import cut_function
from matplotlib.widgets import Slider, RectangleSelector, Button
from scipy.optimize import curve_fit


def change_frontsize(axes,frontsize):
    for ax in axes:
        ax.tick_params(axis="x", labelsize=frontsize)
        ax.tick_params(axis="y", labelsize=frontsize)


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def get_error(f,p,err):
    x = [np.random.normal(p[i], err[i], 10000) for i in range(len(p))]
    y = f(x)
    n, bins = np.histogram(y, 50)
    center = (bins[:-1] + bins[1:]) / 2
    popt, pcov = curve_fit(gaussian, center, n)
    return popt[2]


def double_gaussian(x, amp1, center1, sigma1, amp2, center2, sigma2):
    gaussian1 = amp1 * np.exp(-(x - center1)**2 / (2 * sigma1**2))
    gaussian2 = amp2 * np.exp(-(x - center2)**2 / (2 * sigma2**2))
    return gaussian1 + gaussian2


gaussian_dict = {'single' : gaussian, 'double' : double_gaussian}


def f_tv(x, a, b, c):
    return np.exp((x - a) / b) * c


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
    #risetime.plot_scatter_with_gauss(E[E<20],riset[E<20],ax,len(E)//30)
    scat = ax.scatter(E,1000*riset,s=0.1)
    ax.set_ylim(-20,60)
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylabel('Rise time in ms')
    ax.set_title('Energy vs Rise Time')
    return scat


def plot_decay(E,decayt,ax):
    ax.scatter(E, decayt, s=0.1)
    ax.set_ylim(0,0.1)
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylabel('Decay time in s')
    ax.set_title('Energy vs Decay Time')
def plot_corr(E,correlation,para,ax):
    scat = ax.scatter(E, correlation, s=0.1)
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


class Clicker:
    def __init__(self,x,y):
        self.xdata = x
        self.ydata = y

def Discrimination_power_func(x):
    return np.abs(x[0]-x[2])/np.sqrt(x[1]**2/x[3]**2)

def update_gaussian_fit(gaussian_type_fit, text_parameters,n,center):
    if gaussian_type_fit == 'single':
        popt, pcov = curve_fit(gaussian_dict[gaussian_type_fit], center, n, [n.max(), center.mean(), np.std(center)])
        text_parameters.set_text("$LY_0$ = {:.2e} \n$\sigma$ = {:.2e}".format(popt[1], popt[2]))
        discrimination_power = None
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
        p_error = np.sqrt(np.diag(pcov))
        sel=[False,True,True,False,True,True]

        discrimination_power = Discrimination_power_func(popt[sel])
        discrimination_power_error = get_error(Discrimination_power_func,popt[sel],p_error[sel])
        text_parameters.set_text(
            "$LY_01$ = {:.2e} $\pm$ {:.0e} \n$\sigma1$ = {:.2e} $\pm$ {:.0e}\n$LY_02$ = {:.2e} $\pm$ {:.0e}"
            " \n$\sigma2$ = {:.2e} $\pm$ {:.0e}\nDiscrimination Power = {:.2e} $\pm$ {:.0e}".format(
                popt[1],p_error[1], popt[2],p_error[2], popt[4],p_error[4], popt[5], p_error[5],
                discrimination_power,discrimination_power_error))
    return discrimination_power, popt, pcov

def plot_LY(E,LY,ax, axfit, dictionary):
    clic = Clicker(3500,-0.2)
    release = Clicker(6500,0.6)
    path = dictionary['path']
    nbins = 100
    fig = ax.get_figure()
    pts = ax.scatter(E,LY,s=0.1)
    ax.set_ylabel('Light Yield in keV/MeV')
    ax.set_xlabel('Heat amplitude in keV')
    ax.set_xlim(-100,7000)
    ax.set_title('Heat Energy vs Light Yield for {}'.format(path.split("/")[-2]))
    ax.set_ylim(np.percentile(LY, 5),np.percentile(LY, 80))
    ax_slider = fig.add_axes([0.92, 0.20, 0.01, 0.6])
    slider = Slider(ax_slider, 'bins', valmin=10, valmax=1000, valinit=nbins, valstep=1, orientation="vertical")
    text_parameters = ax.text(0.5, 0.93, 'No parameters yet', fontsize=12, bbox=dict(facecolor='white', edgecolor='white'),color='black', transform=ax.transAxes,verticalalignment='top')
    gaussian_type_fit = 'double'
    # Create a box selection event handler
    discrimination_power = None
    popt, pcov = None, None
    def line_select_callback(eclick, erelease):
        nonlocal discrimination_power
        nonlocal popt
        nonlocal pcov
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        #print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        toggle_selector.RS.set_active(True)
        axfit.clear()
        bool_LY= np.logical_and(LY<np.max((y1,y2)),LY>np.min((y1,y2)))
        bool_E = np.logical_and(E<np.max((x1,x2)),E>np.min((x1,x2)))
        selected_LY = LY[np.logical_and(bool_LY,bool_E)]
        nonlocal nbins

        n, bins = np.histogram(selected_LY, nbins)
        center = (bins[:-1] + bins[1:]) / 2
        hist, = axfit.plot(n ,center, linewidth=.5, ds='steps-mid')
        datax = np.linspace(selected_LY.min(),selected_LY.max(),1000)
        discrimination_power, popt, pcov = update_gaussian_fit(gaussian_type_fit, text_parameters, n, center)
        plot, = axfit.plot(gaussian_dict[gaussian_type_fit](datax, *popt),datax)



        def update(_):
            nonlocal discrimination_power
            nonlocal nbins
            nonlocal popt
            nonlocal pcov
            nbins = int(slider.val)
            n, bins = np.histogram(selected_LY, nbins)
            center = (bins[:-1] + bins[1:]) / 2
            discrimination_power, popt, pcov = update_gaussian_fit(gaussian_type_fit, text_parameters, n, center)
            plot.set_xdata(gaussian_dict[gaussian_type_fit](datax,*popt))
            hist.set_xdata(n)
            hist.set_ydata(center)
            axfit.set_xlim(0,n.max()*1.01)
            fig.canvas.draw_idle()

        #line_select_callback(clic, release)

        slider.on_changed(update)
        fig.canvas.draw_idle()
    def toggle_selector(event):
        nonlocal gaussian_type_fit
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
        if event.key in ['D','d']:
            gaussian_type_fit = 'double'
            print("gaussian type : "+gaussian_type_fit)
        if event.key in ['O','o']:
            gaussian_type_fit = 'single'
            print("gaussian type : " + gaussian_type_fit)

    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    def save_para(_):
        nonlocal discrimination_power
        nonlocal popt
        nonlocal pcov
        nonlocal gaussian_type_fit
        nonlocal dictionary
        path = dictionary['path']
        dictionary_handler.update_dict(path + 'dictionary.json',{'discrimination_power': discrimination_power})
        try:
            dictionary_handler.update_dict(path + 'dictionary.json', {'paramater_LY_'+gaussian_type_fit: popt.tolist(),
                                                                 'uncertainty_LY_'+gaussian_type_fit:
                                                                     (np.diag(pcov)**0.5).tolist()})
        except AttributeError:
            pass
        print('discrimination power saved')


    save_button_ax = fig.add_axes([0.92, 0.05, 0.05, 0.08]) # Define button axes coordinates
    save_button = Button(save_button_ax, 'Save')
    save_button.on_clicked(save_para)


def plot_correlation_heat_light(correlation, correlationl, ax):
    ax.scatter(correlation, correlationl, s=0.1)


def plot(i,j):
    path, dictionary = get_data.get_path(i,j)
    if 'filename' in dictionary:
        E, correlation, amplitude, rt, time_array = get_data.get_pulses(dictionary, ['Energy', 'Correlation', 'Amplitude_filtered',
                                                                         'Rise_time', 'Trigger_position'])
        try:
            para_correlation = dictionary['correlation']
            selection = correlation > cut_function(amplitude, *para_correlation)
        except KeyError:
            print('no correlation cut found')
            selection = np.ones_like(E, dtype=bool)
        E_sel = E[selection]

        #get_data.show_time(ax_rt, scat_rt, time_array, dictionary, type='heat')

    if 'filename_light' in dictionary:
        ampl, rtl = get_data.get_pulses(dictionary, keys=['Energy', 'Rise_time'], type='light')
        '''fig_rt, ax_rt = plt.subplots()
        scat_rt = plot_rise(E, rt, ax_rt)'''
        '''sel_ene = np.logical_and(ampl < 300, ampl > 200)
        sel_rt = np.logical_and(rtl < 0.5, rtl > 0)
        print(np.mean(rtl[np.logical_and(sel_ene, sel_rt)]))'''
    if 'filename_trigheat' in dictionary:
        amplt = get_data.get_pulses(dictionary, keys=['Energy'], type='trigheat')[0]
        LY = amplt[selection] / E_sel * 1000  # in keV/MeV
        fig_LY = plt.figure(figsize=(12, 6))
        # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
        # the size of the marginal axes and the main axes in both directions.
        # Also adjust the subplot parameters for a square plot.
        gs = fig_LY.add_gridspec(1, 2, width_ratios=(4, 1),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.05, hspace=0.05)
        # Create the Axes.
        ax_LY = fig_LY.add_subplot(gs[0, 0])
        axfit = fig_LY.add_subplot(gs[0, 1], sharey=ax_LY)

        axfit.tick_params(left=False, right=False, labelleft=False,
                          labelbottom=False, bottom=False)
        plot_LY(E_sel, LY, ax_LY, axfit, dictionary)
        '''fig_hvl, ax_hvsl = plt.subplots()
        plot_light(E_sel,amplt[selection],ax_hvsl)'''
    print(len(E_sel))



if __name__ == '__main__':
    import dictionary_handler
    import get_data
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
    plot(11,7)
    plot(11, 11)
    plt.show()