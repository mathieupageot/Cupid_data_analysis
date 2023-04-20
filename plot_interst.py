import numpy as np
import matplotlib.pyplot as plt
import lasso_selection
import get_data



path = '/Users/mp274748/Documents/'
peaks = get_data.ntd_array(path+'20180709_23h07.BINLMO.ntp')


print(np.shape(peaks))
correlation = peaks[:,5]
good = correlation > 0.997
amp = peaks[good,2]

t = peaks[good,0]/2000

correlation = peaks[good,5]

baseline = peaks[good,3]

fig3, axs3= plt.subplots(1)
fig2, axs2= plt.subplots(2)
fig,axs=plt.subplots(2,2)
# Pulse energy vs Time raw data heat channel
axs[0,0].scatter(t,amp,s=0.1)
axs[0,0].set_title('Pulse energy vs time raw data heat channel')
axs[0,0].set_xlabel('Time in second')
axs[0,0].set_ylabel('Pulse energy in arbitrary unit')
# Correlation vs Pulse energy raw data heat channel
axs[0,1].scatter(amp,correlation,s=0.1)
axs[0,1].set_title('Correlation vs Pulse energy raw data heat channel')
axs[0,1].set_ylabel('Correlation normalized')
axs[0,1].set_xlabel('Pulse energy in arbitrary unit')
# Pulse energy vs Baseline raw data heat channel: Heater Event
pts=axs[1,0].scatter(baseline,amp,s=0.1)
axs[1,0].set_title('Pulse energy vs Baseline raw data heat channel: Heater Event')
axs[1,0].set_xlabel('Baseline in arbitrary unit')
axs[1,0].set_ylabel('Pulse energy in arbitrary unit')




def accept(event):
    if event.key == "enter":
        selector.disconnect()
        if len(selector.xys[selector.ind][:, 0])>0 :
            sel_baseline = selector.xys[selector.ind][:, 0]
            sel_amp = selector.xys[selector.ind][:, 1]
            axs[1,0].scatter(sel_baseline,sel_amp, s=0.1)
            z=np.polyfit(sel_baseline,sel_amp,1)
            p = np.poly1d(z)
            x = np.linspace(np.min(baseline), np.max(baseline), 1000)
            y = p(x)
            axs[1, 0].plot(x, y, c='r')
            amp_stab = amp / (p(baseline)) * np.mean(sel_amp)
            np.save(path+'amp_stab.npy',amp_stab)

        else :
            amp_stab = np.load(path+'amp_stab.npy')

        axs[1, 1].scatter(baseline, amp_stab, s=0.1)
        axs[1, 1].set_title('Pulse energy vs Baseline raw data heat channel: Heater Event')
        axs[1, 1].set_xlabel('Baseline in arbitrary unit')
        axs[1, 1].set_ylabel('Pulse energy in arbitrary unit')
        fig.canvas.draw()
        p=np.poly1d([ 1.41014879,12.29141961])
        amp_stab_fit=p(amp_stab)


        axs2[1].hist(amp_stab_fit,3000,facecolor='r',label='stabilized data',alpha=0.5)
        axs2[1].set_xlabel('Pulse energy in KeV')
        axs2[1].set_ylabel('Number of events')
        fig2.canvas.draw()
        axs3.scatter(amp_stab_fit[good2], ampl[good2]*100/bin_center[i_max], s=0.1)
        fig3.canvas.draw()


fig.canvas.mpl_connect("key_press_event", accept)
selector = lasso_selection.SelectFromCollection(axs[1,0], pts)
# light

peaksl = get_data.ntd_array(path+'20180709_23h07.BINLD_trigheat.ntp')

tl = peaksl[good,0]
ampl = peaksl[good,2]

correlationl = peaksl[good,5]

baselinel = peaksl[good,3]
good2=correlationl>0.99


axs3.set_ylabel('Light amplitude')
axs3.set_xlabel('Heat amplitude')
axs3.set_title('Heat amplitude VS Light amplitude for alpha discrimination')


peakslt=get_data.ntd_array('/Users/mp274748/Documents/20180709_23h07.BINLD.ntp')
corlt=peakslt[:,5]
goodlt = corlt > 0.99
print(peakslt.shape)
amplt=peakslt[goodlt,2]
n, bins = np.histogram(amplt,600)
bin_center=(bins[:-1]+bins[1:])/2
i_max=np.argmax(n[30:])+30
bin_center[i_max]
amplt_fit = amplt*100/bin_center[i_max]
axs2[0].hist(amplt_fit,600,facecolor='b',label='RAW data')
axs2[0].set_xlabel('Pulse energy in keV')
axs2[0].set_ylabel('Number of events')
#axs3.set_xlim(0)
#axs3.set_ylim(0)

plt.show()
