import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
import matplotlib.pylab as pylab
params = {'legend.fontsize': 20.,
              'figure.figsize': (15, 5),
              'axes.labelsize': 20.,
              'axes.titlesize': 20.,
              'xtick.labelsize': 20.,
              'ytick.labelsize': 20.}
pylab.rcParams.update(params)
def decode_binary(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    return np.frombuffer(binary_data, dtype=np.float64)
fig3,ax3 = plt.subplots()
fig2, ax2 = plt.subplots()
fig,ax = plt.subplots()
def plot_pulse_spectrum(xp,pulse, index):

    sp = np.fft.fft(pulse)

    freq = np.fft.fftfreq(xp.shape[-1])

    ax.plot(freq[freq>=0]*5000, np.absolute(sp)[freq>=0],label="channel {}, mean pulse".format(index))

    return freq[freq>=0]*5000, np.absolute(sp)[freq>=0]
indexes = [1]

for i in indexes:
    path = "/Users/mp274748/Documents/data_arg/RUN96/Measurement6/chan{}/".format(i)
    file_path_spec = path + "000015_20230609T235012_00{}_000.bin_spec.bin".format(i)
    decoded_data = decode_binary(file_path_spec)
    x = np.arange(len(decoded_data))
    ax.plot(x[:len(decoded_data)//2], decoded_data[:len(decoded_data)//2],label="channel {}, noise".format(i))
    file_path_meanpulse = "/Users/mp274748/Documents/data_arg/RUN96/Measurement6/chan{}/000015_20230609T235012_00{}_000.bin_edmean.bin".format(i,i)
    decoded_data = decode_binary(file_path_meanpulse)
    n_mean = len(decoded_data)
    x = np.arange(n_mean)
    xtime=x/5000-0.5
    new_time = np.linspace(-0.1, 0.1, 40000)  # New time values
    interpolated_signal = interp1d(xtime, decoded_data, kind='cubic')(new_time)
    arg_bottom=np.argmin(np.abs(interpolated_signal[new_time<0] - 0.1))
    arg_top = np.argmin(np.abs(interpolated_signal[new_time<0] - 0.9))
    print("Rise Time for channel {}".format(i),new_time[arg_top]-new_time[arg_bottom])
    #ax2.scatter(new_time, interpolated_signal,s=0.1)
    ax2.plot(xtime, decoded_data,label="channel {}, signal".format(i))



    freq, amplitude = plot_pulse_spectrum(x,decoded_data, i)
    ax3.plot(x[:len(decoded_data)//2], amplitude / decoded_data[:len(decoded_data)//2])
    ax3.set_title('signal to noise ratio')
#Rise Time for channel 1 0.0014401440144014288
#Rise Time for channel 2 0.000920092009200929
#ax2.hlines(0.9,-0.5,0.5,label='90%',colors='r')
#ax2.hlines(0.1,-0.5,0.5, label='10%',colors='r')


# Add labels and title to the plot
ax.set_xscale('log')
ax.set_yscale('log')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax2.set_xlabel('time in s')
ax2.set_ylabel('Value')
ax.set_xlabel('frequency')
ax.set_ylabel('Amplitude')
ax.set_title('Noise and Mean pulse spectrum')
fig.legend()
fig2.legend()
# Show the plot
plt.show()