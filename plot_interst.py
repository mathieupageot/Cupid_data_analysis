import numpy as np
import matplotlib.pyplot as plt
import lasso_selection
path = ''
peaks = np.loadtxt(path)
t = peaks[:,]
amp = peaks[:,]
correlation = peaks[:, ]
baseline = peaks[:, ]


fig,axs=plt.subplots(2)
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
pts=axs[1,0].scatter(t,amp)
axs[1,0].set_title('Pulse energy vs Baseline raw data heat channel: Heater Event')
axs[1,0].set_xlabel('Baseline in arbitrary unit')
axs[1,0].set_ylabel('Pulse energy in arbitrary unit')
fig.canvas.mpl_connect("key_press_event", lasso_selection.accept)
selector = lasso_selection.SelectFromCollection(axs[1,0], pts)
def accept(event):
    if event.key == "enter":
        #print("Selected points:")
        #print(selector.xys[selector.ind])
        selector.disconnect()
        ax[1,1].set_title("")
        axs[1,1].scatter(selector.xys[selector.ind][:, 0], selector.xys[selector.ind][:, 1], s=80)
        fig.canvas.draw()



