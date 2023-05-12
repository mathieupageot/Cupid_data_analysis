import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

import get_data
path, filename, filename_light, filename_trigheat,filenum= get_data.get_path()
peaks = get_data.ntd_array(path+filename)
amp= np.load(path+'amp_stab.npy')
correlation = peaks[:,5]
Riset=peaks[:, 11]
correl_cut=0.9998
Rise_cut=0.25
good = np.logical_and(correlation>correl_cut,Riset<Rise_cut)


amp=amp[good]
fig3,ax3 = plt.subplots()
fig2, ax2 = plt.subplots()
fig,ax=plt.subplots()
fig.subplots_adjust(left=0.25)

n , bins = np.histogram(amp, 1000)
center= (bins[:-1] + bins[1:]) / 2
if filenum == 2:
    data_E=np.array([352,609,295,242])
    data_amp=np.array([1114,640,537,433])
if filenum == 3:
    data_E = np.array([352,609,768,1120,1238,1764])
    data_amp = np.array([145,319,502,246,460,714])
data_E=np.sort(data_E)
data_amp=np.sort(data_amp)
lines=[]
z = np.polyfit(data_amp,data_E,1)
#z  = [1,0] #4800/9000
p = np.poly1d(z)

n2, bins2 = np.histogram(p(amp)[:6000], 1000)
center2= (bins2[:-1] + bins2[1:]) / 2

for i in range(len(data_amp)):
    line,=ax.plot([data_amp[i],data_amp[i]],[0,n.max()], label=str(data_E[i])+' keV',alpha=0.4,linewidth=.5)
    #line2, = ax3.plot([p(data_amp[i]), p(data_amp[i])], [0, n2.max()], label=str(np.round(p(data_amp[i]),1)) + ' keV', alpha=0.4, linewidth=.5)
    line2, = ax3.plot([data_E[i],data_E[i]], [0, n2.max()], label=str(np.round(data_E[i],1)) + ' keV', alpha=0.4, linewidth=3)

    lines.append(line)
    ax3.legend()
    ax.legend()

scat=ax2.scatter(data_amp,data_E)
x = np.linspace(min(data_amp) - max(data_amp) * 0.1, max(data_amp) * 1.1, 700)
y = p(x)
line, = ax2.plot(x, y, c='r')

hist3,=ax3.plot(center2,n2,linewidth=.5,ds='steps-mid')
ax3.set_title('Heat Channel Spectrum Calibrated')
ax3.set_xlabel('Energy in keV')
ax3.set_ylabel('Counts/6 keV')
hist,=ax.plot(center,n,linewidth=.5,ds='steps-mid')
axbin = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
        ax=axbin,
        label="Bins",
        valmin=300,
        valmax=5000,
        valinit=1000,
        orientation="vertical",
        valstep=1.
    )
def update(val):
    n, bins = np.histogram(amp, int(val))
    center = (bins[1:] + bins[:-1]) / 2
    hist.set_xdata(center)
    hist.set_ydata(n)
    fig.canvas.draw()
    for i,line in enumerate(lines):
        line.set_ydata([0,n.max()])
    ax.set_ylim(0,n.max())
amp_slider.on_changed(update)
print(z)
plt.show()
