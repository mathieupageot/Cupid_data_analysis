import numpy as np
import matplotlib.pyplot as plt
import lasso_selection
import get_data
from scipy.optimize import curve_fit
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


path,filename, filename_light, filename_trigheat = get_data.get_path()
peaks = get_data.ntd_array(path+filename)
correlation = peaks[:,5]
amp = peaks[:,2]
t = peaks[:,0]/5000
baseline = peaks[:,3]
fig,axs = plt.subplots(2)
def accept(event):
    if event.key == "enter":
        selector.disconnect()
        sel_baseline = selector.xys[selector.ind][:, 0]
        sel_amp = selector.xys[selector.ind][:, 1]
        axs[0].scatter(sel_baseline,sel_amp, s=0.1)
        z=np.polyfit(sel_baseline,sel_amp,1)
        p = np.poly1d(z)
        x = np.linspace(np.min(baseline), np.max(baseline), 1000)
        y = p(x)
        axs[0].plot(x, y, c='r')
        amp_stab = amp / (p(baseline)) * np.mean(sel_amp)
        axs[1].scatter(baseline, amp_stab, s=0.1)
        axs[1].set_ylabel('Heat Amplitude Stabilized')
        axs[1].set_xlabel('Heat Baseline Stabilized')
        np.save(path+'amp_stab.npy',amp_stab)
pts=axs[0].scatter(baseline,amp,s=0.1)
axs[0].set_ylabel('Heat Amplitude Raw')
axs[0].set_xlabel('Heat Baseline Raw')
fig.canvas.mpl_connect("key_press_event", accept)
selector = lasso_selection.SelectFromCollection(axs[0], pts)
plt.show()
