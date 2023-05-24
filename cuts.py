import numpy as np
import matplotlib.pyplot as plt
import lasso_selection
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

import get_data
from scipy.optimize import curve_fit

path, filename, filename_light, filename_trigheat, p = get_data.get_path()
peaks = get_data.ntd_array(path + filename)
amp_stab = np.load(path + 'amp_stab.npy')
amp_stab_fit = p(amp_stab)
correlation = peaks[:, 5]
t = peaks[:, 0] / 5000
TV = peaks[:, 8]  # parabolic cut
Riset = peaks[:, 11]
decayt = peaks[:, 12]
X = []
Y = []
cut_name = 'correlation'




def f(x, a, b, c):
    return a * x ** b + c


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.30, bottom=0.30)

ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'

axa = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axb = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axc = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)

if cut_name == 'TV':
    a0=2
    b0=1e6
    c0=1
    cut = f(amp_stab_fit, a0, b0, c0) > TV
    pts = ax.scatter(amp_stab_fit, TV, s=0.1)
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylabel('TV')
    ax.set_title('Energy vs TV')
elif cut_name == 'correlation':
    a0 = -1
    b0 = -1
    c0 = 1
    cut = f(amp_stab_fit, a0, b0, c0) > correlation
    pts = ax.scatter(amp_stab_fit, correlation, s=0.1)
    ax.set_xlabel('Pulse energy in keV')
    ax.set_ylabel('correlation')
    ax.set_title('Energy vs correlation')
    ax.set_ylim(0.99,1)

sa = Slider(axa, 'a', -100, 100, valinit=a0)
sb = Slider(axb, 'b', -2, 2, valinit=b0)
sc = Slider(axc, 'Curvature', 0.99, 1.01, valinit=c0)
amp = amp_stab_fit[cut]
nTV, binsTV = np.histogram(amp[amp < 6000], 3000)
centerTV = (binsTV[:-1] + binsTV[1:]) / 2
fig2, axs2 = plt.subplots()
hist, = axs2.plot(centerTV, nTV, linewidth=.5, ds='steps-mid', label='TV cut')
fig2.subplots_adjust(bottom=0.2)
a,b,c=a0, b0, c0
def update(val):
    a = sa.val
    b = sb.val
    c = sc.val
    data = f(xp, a,b,c)
    l.set_ydata(data)
    if cut_name == 'TV':
        cut = f(amp_stab_fit, a, b, c) > TV
    elif cut_name == 'correlation':
        cut = f(amp_stab_fit, a, b, c) > correlation
    amp = amp_stab_fit[cut]

    nTV2, binsTV = np.histogram(amp, 3000)

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

    np.save(path+filename.strip("ntp")+"npy",[a,b,c])
    print(a,b,c)

sa.on_changed(update)
sb.on_changed(update)
sc.on_changed(update)
xp = np.linspace(0.1, amp_stab_fit.max(), 10000)
l, = ax.plot(xp, f(xp, a0, b0, c0),c='r')
axprev = fig2.add_axes([0.7, 0.05, 0.1, 0.075])
bprev = Button(axprev, 'Save')

bprev.on_clicked(save)


plt.show()
