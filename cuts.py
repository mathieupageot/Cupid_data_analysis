import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

import dictionary_handler
import get_data


def cut_function(x, a, b, c, x0=0):
    positive = x>0
    y=np.ones_like(x)*2
    y[positive] = a * np.power(x[positive]-x0, b) + c
    return y

def plot_cutTV(ax,TV,amp,a0,b0,c0):
    cut = cut_function(amp, a0, b0, c0) > TV
    pts = ax.scatter(amp, TV, s=0.1)
    ax.set_xlabel('Amplitude in arbitrary unit')
    ax.set_ylabel('TV')
    ax.set_title('Amplitude vs TV')
    sa = Slider(axa, 'a', 0, 0.001, valinit=a0)
    sb = Slider(axb, 'b', 0, 4, valinit=b0)
    sc = Slider(axc, 'c', 0, 1000, valinit=c0)
    s0 = Slider(axc, 'x0', -10., 100., valinit=c0)
    return sa, sb, sc,s0, cut, pts

def plot_cutcorr(ax,correlation,amp,a0,b0,c0,x0,axa,axb,axc,ax0):
    cut = cut_function(amp, a0, b0, c0, x0) < correlation
    pts = ax.scatter(amp, correlation, s=0.1)
    ax.set_xlabel('Amplitude in arbitrary unit')
    ax.set_ylabel('Correlation')
    ax.set_title('Energy vs correlation')
    ax.set_ylim(0.90,1)
    sa = Slider(axa, 'a', 2*a0, 0, valinit=a0)
    sb = Slider(axb, 'b', 2*b0, 0, valinit=b0)
    sc = Slider(axc, 'c', 0.9, 1.1, valinit=c0)
    s0 = Slider(ax0, 'x0', -5., +5., valinit=x0)
    return sa, sb, sc,s0, cut, pts

def hist_plot(amp,cut,axs2,fig):
    cutted_amp = amp[cut]
    nTV, binsTV = np.histogram(cutted_amp, 3000)
    centerTV = (binsTV[:-1] + binsTV[1:]) / 2
    hist, = axs2.plot(centerTV, nTV, linewidth=.5, ds='steps-mid', label='TV cut')
    fig.subplots_adjust(bottom=0.2)
    #axs2.set_xlim(0,6000)
    return hist


def get_cut_para(amp, para_cut, path, correlation, cut_name ='correlation'):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.30, bottom=0.30)

    ax.margins(x=0)
    axcolor = 'lightgoldenrodyellow'

    axa = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axb = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axc = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax0 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    if cut_name == 'TV':
        a0, b0, c0 = 0.0005, 2, 1
        sa,sb,sc,s0, cut, pts = plot_cutTV(ax,TV,amp,a0,b0,c0)
    elif cut_name == 'correlation':
        a0, b0, c0, x0 = para_cut
        sa,sb,sc, s0, cut, pts = plot_cutcorr(ax,correlation,amp,a0,b0,c0,x0,axa,axb,axc,ax0)
    fig2, axs2 = plt.subplots()
    hist = hist_plot(amp, cut, axs2,fig2)
    def update(val):
        a = sa.val
        b = sb.val
        c = sc.val
        x0 = s0.val
        data = cut_function(xp, a, b, c, x0)
        l.set_ydata(data)
        if cut_name == 'TV':
            cut = cut_function(amp, a, b, c, x0) > TV
        elif cut_name == 'correlation':
            cut = cut_function(amp, a, b, c, x0) < correlation
        nTV2, binsTV = np.histogram(amp[cut], 3000)
        centerTV = (binsTV[:-1] + binsTV[1:]) / 2
        hist.set_xdata(centerTV)
        hist.set_ydata(nTV2)
        axs2.set_ylim(-1, nTV2.max())
        fig.canvas.draw_idle()
        fig2.canvas.draw_idle()
    def save(event):
        a = sa.val
        b = sb.val
        c = sc.val
        x0 = s0.val
        dictionary_handler.update_dict(path + "dictionary.json", {cut_name: [a, b, c, x0]})
        print('cut parameters saved')


    sa.on_changed(update)
    sb.on_changed(update)
    sc.on_changed(update)
    s0.on_changed(update)
    xp = np.linspace(1e-6, amp.max(), 10000)
    l, = ax.plot(xp, cut_function(xp, a0, b0, c0), c='r')
    axprev = fig2.add_axes([0.7, 0.05, 0.1, 0.075])
    bprev = Button(axprev, 'Save')

    bprev.on_clicked(save)
    plt.show()

if __name__=="__main__":
    path, dictionary = get_data.get_path()
    amplitude, correlation = get_data.get_pulses(dictionary,['amp_stab','Correlation'])
    #amplitude, correlation = get_data.get_pulses(dictionary, ['Amplitude_filtered', 'Correlation'], type='light')
    try:
        para_correlation = dictionary['correlation']
        print('cut from dictionary')
    except KeyError:
        para_correlation = np.array([-0.001, -1, 0.95, 1.e-5])
        print('no correlation cut found')
    amp_max = dictionary["mean_pulse_max"]
    get_cut_para(amplitude[amplitude < amp_max], para_correlation, path, correlation[amplitude < amp_max])
