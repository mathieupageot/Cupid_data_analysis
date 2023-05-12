import numpy as np
import matplotlib.pyplot as plt
import lasso_selection
from matplotlib.widgets import Slider

import get_data
from scipy.optimize import curve_fit
path,filename, filename_light, filename_trigheat, filenum = get_data.get_path()
peaks = get_data.ntd_array(path+filename)
amp_stab = np.load(path+'amp_stab.npy')
if filenum == 2:
    z=[0.54076489, 6.23910689]
if filenum == 3:
    z=[  2.47881842, -10.57679343]
else:
    z=[1,0]
p = np.poly1d(z)
amp_stab_fit=p(amp_stab)
correlation = peaks[:,5]
t = peaks[:, 0] / 5000
TV=peaks[:, 8] #parabolic cut
Riset=peaks[:, 11]
decayt=peaks[:, 12]
X=[]
Y=[]
def f_TV(x,ymin,ymax,n,xmax):
    return (ymax-ymin)*(x/xmax)**(2*n)+ymin

fig,ax = plt.subplots()
plt.subplots_adjust(left=0.30, bottom=0.30)

ax.margins(x=0)
axcolor = 'lightgoldenrodyellow'
xmax=73500
axymin = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axymax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axn = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
symin = Slider(axymin, 'ymin', 0, 100, valinit=2)
symax = Slider(axymax, 'ymax', 0, 1.1e6, valinit=1e6)
sn = Slider(axn, 'Curvature', 1, 10, valinit=1,valstep=1)
TV_cut = f_TV(amp_stab_fit,2,1e6,1,xmax)>TV
amp_TV = amp_stab_fit[TV_cut]
nTV, binsTV = np.histogram(amp_TV[amp_TV < 6000], 3000)
centerTV = (binsTV[:-1] + binsTV[1:]) / 2
fig2,axs2=plt.subplots()
hist,=axs2.plot(centerTV, nTV, linewidth=.5, ds='steps-mid', label='TV cut')
def update(val):
    ymax = symax.val
    ymin = symin.val
    n = sn.val
    data=f_TV(xp,ymin,ymax,n,xmax)
    l.set_ydata(data)
    TV_cut=f_TV(amp_stab_fit,ymin,ymax,n,xmax)>TV

    amp_TV = amp_stab_fit[TV_cut]
    print(amp_TV.shape)
    nTV2, binsTV = np.histogram(amp_TV, 3000)


    centerTV = (binsTV[:-1] + binsTV[1:]) / 2

    hist.set_xdata(centerTV)
    hist.set_ydata(nTV2)
    axs2.set_ylim(-1,nTV2.max())

    fig.canvas.draw_idle()
    fig2.canvas.draw_idle()
symin.on_changed(update)
symax.on_changed(update)
sn.on_changed(update)
ymin=TV.min()
xp=np.linspace(0,xmax,10000)
l,=ax.plot(xp,f_TV(xp,2,1e6,1,xmax))
pts=ax.scatter(amp_stab_fit, TV, s=0.1)
ax.set_xlabel('Pulse energy in keV')
ax.set_ylabel('TV')
ax.set_title('Energy vs TV')


plt.show()