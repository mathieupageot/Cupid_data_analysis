import numpy as np
import matplotlib.pyplot as plt
import lasso_selection
import get_data
path = '/Users/mp274748/Documents/20180709_23h07.BINLMO.ntp'
peaks = get_data.ntd_array(path)
t = peaks[:,0]
amp = peaks[:,3]
correlation = peaks[:,5]

baseline = peaks[:,4]


fig,axs=plt.subplots(2,2)
# Pulse energy vs Time raw data heat channel
axs[0,0].scatter(t,amp)
axs[0,0].set_title('Pulse energy vs time raw data heat channel')
axs[0,0].set_xlabel('Time in second')
axs[0,0].set_ylabel('Pulse energy in arbitrary unit')
# Correlation vs Pulse energy raw data heat channel
axs[0,1].scatter(amp,correlation)
axs[0,1].set_title('Correlation vs Pulse energy raw data heat channel')
axs[0,1].set_ylabel('Correlation normalized')
axs[0,1].set_xlabel('Pulse energy in arbitrary unit')
# Pulse energy vs Baseline raw data heat channel: Heater Event
pts=axs[1,0].scatter(baseline,amp)
axs[1,0].set_title('Pulse energy vs Baseline raw data heat channel: Heater Event')
axs[1,0].set_xlabel('Baseline in arbitrary unit')
axs[1,0].set_ylabel('Pulse energy in arbitrary unit')
fig2, axs2= plt.subplots(1)

axs2.hist(amp,50,facecolor='b',label='RAW data')
axs2.set_xlabel('Pulse energy in arbitrary unit')
axs2.set_ylabel('Number of events')
axs2.set_title('Spectrum Comparison')
axs2.legend()
def accept(event):
    if event.key == "enter":
        #print("Selected points:")
        #print(selector.xys[selector.ind])
        selector.disconnect()
        sel_baseline = selector.xys[selector.ind][:, 0]
        sel_amp = selector.xys[selector.ind][:, 1]
        axs[1,0].scatter(sel_baseline,sel_amp, s=80)
        z=np.polyfit(sel_baseline,sel_amp,1)
        p = np.poly1d(z)
        x=np.linspace(np.min(baseline),np.max(baseline),1000)
        y=p(x)
        axs[1, 0].plot(x,y,c='r')
        fig.canvas.draw()
        amp_stab=amp/(p(baseline))*np.mean(sel_amp)
        axs[1, 1].scatter(t, amp_stab, s=80)
        axs[1, 1].set_xlabel('Baseline in arbitrary unit')
        axs[1, 1].set_ylabel('Pulse energy in arbitrary unit')
        axs[1, 1].set_title("Pulse energy stabilized vs Baseline heat channel")


        axs2.hist(amp_stab,50,facecolor='r',label='stabilized data')
        axs2.legend()
        fig2.canvas.draw()


fig.canvas.mpl_connect("key_press_event", accept)
selector = lasso_selection.SelectFromCollection(axs[1,0], pts)

plt.show()
