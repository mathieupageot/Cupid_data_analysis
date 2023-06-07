import numpy as np
import matplotlib.pyplot as plt
import lasso_selection
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

import get_data
from scipy.optimize import curve_fit



def f(x, a, b, c,x0=0):
    return a * (x-x0) ** b + c
def plot_cutTV(ax,TV,amp,a0,b0,c0):
    cut = f(amp, a0, b0, c0) > TV
    pts = ax.scatter(amp, TV, s=0.1)
    ax.set_xlabel('Amplitude in arbitrary unit')
    ax.set_ylabel('TV')
    ax.set_title('Amplitude vs TV')
    sa = Slider(axa, 'a', 0, 0.001, valinit=a0)
    sb = Slider(axb, 'b', 0, 4, valinit=b0)
    sc = Slider(axc, 'c', 0, 1000, valinit=c0)
    s0 = Slider(axc, 'x0', -10., 100., valinit=c0)
    return sa, sb, sc,s0, cut, pts

def plot_cutcorr(ax,correlation,amp,a0,b0,c0,x0):
    cut = f(amp, a0, b0, c0) < correlation
    pts = ax.scatter(amp, correlation, s=0.1)
    ax.set_xlabel('Amplitude in arbitrary unit')
    ax.set_ylabel('Correlation')
    ax.set_title('Energy vs correlation')
    ax.set_ylim(0.99,1)
    sa = Slider(axa, 'a', -10, 0, valinit=a0)
    sb = Slider(axb, 'b', -2, 0, valinit=b0)
    sc = Slider(axc, 'c', 0.9, 1., valinit=c0)
    s0 = Slider(ax0, 'x0', -200., +50., valinit=x0)
    return sa, sb, sc,s0, cut, pts

def hist_plot(amp,cut,axs2):
    cutted_amp = amp[cut]
    nTV, binsTV = np.histogram(cutted_amp, 3000)
    centerTV = (binsTV[:-1] + binsTV[1:]) / 2
    hist, = axs2.plot(centerTV, nTV, linewidth=.5, ds='steps-mid', label='TV cut')
    fig2.subplots_adjust(bottom=0.2)
    axs2.set_xlim(0,6000)
    return hist



if __name__=="__main__":
    path, filename, filename_light, filename_trigheat, p = get_data.get_path()
    peaks = get_data.ntd_array(path + filename)
    amp = np.load(path + 'amp_stab.npy')
    correlation = peaks[:, 5]
    t = peaks[:, 0] / 5000
    TV = peaks[:, 8]  # parabolic cut
    Riset = peaks[:, 11]
    decayt = peaks[:, 12]
    try:
        para_cut = np.load(path+filename.strip(".ntp")+'_'+'correlation'+".npy")
    except FileNotFoundError:
        para_cut = np.array([-1, -1, 0.80,0])
        print('no correlation cut found')
    X = []
    Y = []
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.30, bottom=0.30)

    ax.margins(x=0)
    axcolor = 'lightgoldenrodyellow'

    axa = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axb = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    axc = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax0 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    cut_name = 'correlation'
    #cut_name = 'TV'
    if cut_name == 'TV':
        a0, b0, c0 = 0.0005, 2, 1
        sa,sb,sc,s0, cut, pts = plot_cutTV(ax,TV,amp,a0,b0,c0)
    elif cut_name == 'correlation':
        a0, b0, c0 ,x0 = para_cut
        sa,sb,sc, s0, cut, pts = plot_cutcorr(ax,correlation,amp,a0,b0,c0,x0)
    fig2, axs2 = plt.subplots()
    hist = hist_plot(amp, cut, axs2)
    def update(val):
        a = sa.val
        b = sb.val
        c = sc.val
        x0 = s0.val
        data = f(xp, a, b, c, x0)
        l.set_ydata(data)
        if cut_name == 'TV':
            cut = f(amp, a, b, c, x0) > TV
        elif cut_name == 'correlation':
            cut = f(amp, a, b, c, x0) < correlation
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
        np.save(path + filename.strip(".ntp") + '_' + cut_name + ".npy", [a, b, c,x0])
        print(a, b, c, x0)


    sa.on_changed(update)
    sb.on_changed(update)
    sc.on_changed(update)
    s0.on_changed(update)
    xp = np.linspace(0.1, amp.max(), 10000)
    l, = ax.plot(xp, f(xp, a0, b0, c0), c='r')
    axprev = fig2.add_axes([0.7, 0.05, 0.1, 0.075])
    bprev = Button(axprev, 'Save')

    bprev.on_clicked(save)
    plt.show()
