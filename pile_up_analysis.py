import numpy as np
from get_data import ntd_array
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
def decode_binary(file_path, data_type=np.float64):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    return np.frombuffer(binary_data, dtype=data_type)


files_path = ["/Users/mp274748/Documents/data_arg/RUN96/Measurement6/LMO18/000015_20230609T235012_00{}_000.bin".format(i) for i in
              [1, 8]]
datas_arg_path = ["/Users/mp274748/Documents/data_arg/RUN96/Measurement6/chan{}/000015_20230609T235012_00{}_000.bin.ntp"
                  .format(i, i) for i in [1, 2, 7, 8]]

def get_time_pileup(i,tmin):
    peaks = ntd_array(datas_arg_path[0])
    time = peaks[:,0]
    pileup_times = time[np.argwhere(time[1:]-time[:-1] < tmin)+1]
    return pileup_times.astype(int).reshape(-1)


def update_plot(pileup_time, light_channel, heat_channel, light_plot, heat_plot):
    light_plot.set_ydata(light_channel[pileup_time-2500:pileup_time+2500])
    heat_plot.set_ydata(heat_channel[pileup_time - 2500:pileup_time + 2500])


def show_pileup(pileup_times, files_path):
    fig, ax = plt.subplots()
    light_channel = decode_binary(files_path[0], data_type=np.uint32)
    heat_channel = decode_binary(files_path[1], data_type=np.uint32)
    pileup_num = 0
    pileup_time = pileup_times[pileup_num]
    light_plot, = ax.plot(light_channel[pileup_time-1000:pileup_time+4000])
    heat_plot, = ax.plot(heat_channel[pileup_time-1000:pileup_time+4000])
    plt.subplots_adjust(bottom=0.25)
    button_ax_up = plt.axes([0.8, 0.05, 0.1, 0.05])
    button_up = Button(button_ax_up, 'up')
    button_ax_down = plt.axes([0.7, 0.05, 0.1, 0.05])
    button_down = Button(button_ax_down, 'down')
    def go_down(_):
        nonlocal pileup_num
        nonlocal pileup_time
        if pileup_num > 0 :
            pileup_num -= 1
            update_plot(pileup_times[pileup_num], light_channel, heat_channel, light_plot, heat_plot)
            fig.canvas.draw()
    def go_up(_):
        nonlocal pileup_num
        nonlocal pileup_time
        if pileup_num < len(pileup_times)-1 :
            pileup_num += 1
            update_plot(pileup_times[pileup_num], light_channel, heat_channel, light_plot, heat_plot)
            fig.canvas.draw()

    button_up.on_clicked(go_up)
    button_down.on_clicked(go_down)
    fig.canvas.mpl_connect("key_press_event", go_up)

if __name__ == '__main__':
    pileup_times = get_time_pileup(1, 2500)
    print(len(pileup_times))
    show_pileup(pileup_times, files_path)
    plt.show()



