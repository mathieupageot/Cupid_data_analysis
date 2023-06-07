import numpy as np
import matplotlib.pyplot as plt
import get_data
from landaupy import langauss
import gaussian_fit_light
from cuts import f
from matplotlib.widgets import Slider, RectangleSelector
import lasso_selection
from scipy.optimize import curve_fit
def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))



def f_tv(x, a, b, c):
    return np.exp((x - a) / b) * c
def picker(fig):
    def onpick(event):
        ind = event.ind
        print('onpick scatter:', time[ind]-3000)
    fig.canvas.mpl_connect('pick_event', onpick)
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


def plot_rise(E,riset,pick=0.3):
    fig5, ax5 = plt.subplots(1)
    ax5.scatter(E, riset, s=0.1,picker=pick)

    ax5.set_ylim(-0.02,0.06)
    ax5.set_xlabel('Pulse energy in keV')
    ax5.set_ylabel('Rise time in s')
    ax5.set_title('Energy vs Rise Time')
    if pick:
        picker(fig5)
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
def hist_light_fit(amplt,filename):
    fig4, axs4 = plt.subplots(1)
    popt, pcov, hist, bin_centers = gaussian_fit_light.binned_fit_langauss(amplt[amplt > 3400], bins='auto',
                                                                           nan='remove')
    axs4.plot(bin_centers, hist, linewidth=.5, ds='steps-mid')
    x_plot = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    axs4.plot(x_plot, langauss.pdf(x_plot, *popt))
    axs4.set_xlabel("Light amplitude in arbitrary unit")
    axs4.set_ylabel("Counts/" + str(int((bin_centers[-1] - bin_centers[0]) / len(bin_centers)) + 1) + ' arbitrary units')
    axs4.set_title('Landeau fit of the light channel')

    if filename == '20180709_23h07.BINLD.ntp':
        amplt_fit = amplt * 100 / popt[0]
        ampl_fit = ampl * 100 / popt[0]
        Emu = 100
    elif filename == '20211125_00h43.BINLD21.2.ntp':
        amplt_fit = amplt * 260 / popt[0]
        ampl_fit = ampl * 260 / popt[0]
        Emu = 260
    else:
        amplt_fit = amplt * 100 / popt[0]
        ampl_fit = ampl * 100 / popt[0]
        Emu = 100
    print("keV/ADU for light: {:.2e} & error: {:.2e}".format(Emu / popt[0], np.sqrt(pcov[0, 0]) * Emu / popt[0] ** 2))
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
def plot_LY(E,LY):
    fig11 = plt.figure()
    ax11 = fig11.add_subplot(111)
    left, bottom, width, height = 0.75, 0.75, 0.2, 0.2
    ax12 = fig11.add_axes([left, bottom, width, height])
    pts = ax11.scatter(E,LY,s=0.1)
    ax11.set_ylabel('LY ADU')
    ax11.set_xlabel('Heat amplitude in keV')
    ax11.set_title('Heat Energy vs LY')
    ymean=np.mean(LY)
    yvar=np.var(LY)
    ax11.set_ylim(ymean-yvar,ymean+yvar)
    ax_slider = plt.axes([0.75, 0.65, 0.2, 0.03])
    slider = Slider(ax_slider, 'bins', valmin=10, valmax=1000, valinit=100, valstep=1)
    text_obj = ax11.text(0.8, 0.60, 'No parameters yet', fontsize=12, color='red', transform=ax11.transAxes)
    # Create a box selection event handler
    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        toggle_selector.RS.set_active(True)
        ax12.clear()
        bool_LY= np.logical_and(LY<np.max((y1,y2)),LY>np.min((y1,y2)))
        bool_E = np.logical_and(E<np.max((x1,x2)),E>np.min((x1,x2)))
        selected_LY = LY[np.logical_and(bool_LY,bool_E)]
        nbins = int(len(selected_LY)/50)+1
        slider.valmin, slider.valmax, slider.valinit = nbins*0.1, nbins*10, nbins

        n, bins = np.histogram(selected_LY, nbins)
        center = (bins[:-1] + bins[1:]) / 2
        hist, = ax12.plot(center, n, linewidth=.5, ds='steps-mid')
        datax = np.linspace(selected_LY.min(),selected_LY.max(),1000)
        popt, pcov = curve_fit(gauss, center, n, [n.max(), center.mean(), np.std(center)])
        plot, = ax12.plot(datax,gauss(datax, *popt))
        text_obj.set_text("$LY_0$ = {:.2e} \n$\sigma$ = {:.2e}".format(popt[1], popt[2]))



        def update(_):
            nbins = int(slider.val)
            n, bins = np.histogram(selected_LY, nbins)
            center = (bins[:-1] + bins[1:]) / 2
            popt, pcov = curve_fit(gauss,center,n,[n.max(), center.mean(), np.std(center)])
            plot.set_ydata(gauss(datax,*popt))
            text_obj.set_text("$LY_0$ = {:.2e} \n$\sigma$ = {:.2e}".format(popt[1], popt[2]))
            hist.set_xdata(center)
            hist.set_ydata(n)
            ax12.set_ylim(0,n.max()*1.01)
            fig11.canvas.draw_idle()

        slider.on_changed(update)
        fig11.canvas.draw_idle()
    def toggle_selector(event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')

    toggle_selector.RS = RectangleSelector(ax11, line_select_callback,
                                           useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    selector = lasso_selection.SelectFromCollection(ax11, pts)

if __name__ == '__main__':
    path, filename, filename_light, filename_trigheat, p= get_data.get_path()
    E, amp, correlation, TV, riset, decayt, Sm, time = get_data.get_heat(path, filename, p)
    try:
        para_corr = np.load(path+filename.strip(".ntp")+'_'+'correlation'+".npy")
    except FileNotFoundError:
        print('No correlation cut found')
        para_corr = np.array([-1,-1,0.80])
    correl_cut = correlation > f(amp,*para_corr)
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
        amplt_fit, ampl_fit = hist_light_fit(amplt,filename_light)
        LY = ampl_fit / E_sel
        plot_light(E_sel, ampl_fit)
        plot_LY(E_sel, LY)

    plt.show()