import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

import get_data
path, filename, filename_light, filename_trigheat = get_data.get_path()
amp= np.load(path+'amp_stab.npy')

fig,ax=plt.subplots()
fig.subplots_adjust(bottom=0.4)
fig2, ax2 = plt.subplots()
n, bins, patches = ax.hist(amp, 1000)
gs = gridspec.GridSpec(4, 4,figure=fig)
gs.update(left=0.1, right=0.9, bottom=0.1, top=0.2, hspace=0.1, wspace=0.5)
data=np.zeros(16)
lines=[]
for i in range(8):
    line,=ax.plot([data[i*2],data[i*2]],[0,n.max()*1.1], label=i,alpha=0.5)
    lines.append(line)
z = [0,0]
p = np.poly1d(z)
scat=ax2.scatter(data[::2], data[1::2])
x = np.linspace(min(data[::2]) - max(data[::2]) * 0.1, max(data[::2]) * 1.1, 700)
y = p(x)
line, = ax2.plot(x, y, c='r')


axes = [fig.add_subplot(gs[i, j]) for i, j in [(i, j) for i in range(4) for j in range(4)]]
T = []
for i in range(4):
    for j in range(2):
        T.append(
            TextBox(axes[j*2 + i * 4], 'Amp ' + str(j*2 + i * 4), initial=str(0.), hovercolor='0.975', label_pad=0.1))
        T.append(
            TextBox(axes[j*2 + i * 4 + 1], 'E ' + str(j*2 + i * 4), initial=str(0.), hovercolor='0.975', label_pad=0.1))

def submit(val):
    data=np.array([float(tb.text) for tb in T])
    print(data)
    for i in range(8):
        lines[i].set_xdata([data[i*2],data[i*2]])
    x_data=data[::2]
    print(x_data)
    y_data=data[1::2]
    sel = x_data > 0.
    print(x_data[sel])
    if len(x_data[sel])>1:


        z = np.polyfit(x_data[sel], y_data[sel], 1)
        p = np.poly1d(z)
        scat.set_offsets(np.array([data[::2], data[1::2]]).T)
        x = np.linspace(min(data[::2]) - max(data[::2]) * 0.1, max(data[::2]) * 1.1, 700)
        y = p(x)
        line.set_xdata(x)
        line.set_ydata(y)
        ax2.set_xlim(mix(x),max(x))
        ax2.set_ylim(mix(y), max(y))


    fig.canvas.draw_idle()
    fig2.canvas.draw()
axbin = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
        ax=axbin,
        label="Bins",
        valmin=500,
        valmax=5000,
        valinit=700,
        orientation="vertical"
    )
def update(val):
    n, _ = np.histogram(amp, val)
    for count, rect in zip(n, patches.patches):
        rect.set_height(count)
    fig.canvas.draw_idle()
amp_slider.on_changed(update)
for tb in T:
    tb.on_submit(submit)

plt.show()
