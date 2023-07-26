import matplotlib.pyplot as plt
import numpy as np
import datetime
from utility.display_bin import plot_window, decode_binary
from dictionary_handler import load_dict, update_dict, get_values
import json


def save_non_homog_list(non_homogeneous_array,non_homogeneous_array2,path):
    with open(path, 'w') as file:
        json.dump(non_homogeneous_array, file)
        json.dump(non_homogeneous_array2, file)


def function_stabilize(amp, baseline, stab_param, mean_value):
    p = np.poly1d(stab_param)
    return amp / (p(baseline)) * mean_value

def find_local_maxima(data):
    local_maxima = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            local_maxima.append(data[i])
    local_maxima.append(data[-1])
    return local_maxima


def get_duration(path, dictionary):
    filename = dictionary["filename"]
    try:
        peaks = ntd_array(path + filename)
    except TypeError:
        filename = dictionary["filename_light"]
        peaks = ntd_array(path + filename)
    duration = np.sum(find_local_maxima(peaks[:, 0]))
    update_dict(path + "dictionary.json", {"duration": duration})
    return duration


def ntd_array(path):
    A = np.loadtxt(path, usecols=(n for n in range(20)))
    return A


def get_time(fig, scat, time_array):
    scat.set_picker(5)

    def onpick(event):
        ind = event.ind
        print('onpick scatter:', time_array[ind] - 5000)

    fig.canvas.mpl_connect('pick_event', onpick)


def show_time(ax, scat, time_array, dictionary, type='heat'):
    fig = ax.get_figure()
    scat.set_picker(5)
    fig_display, ax_display = plt.subplots()
    plot_display, = ax_display.plot(np.zeros(1000))
    path = dictionary["path"]
    try:
        filename = dictionary["filename_" + type]
    except KeyError:
        filename = dictionary["filename"]
    file_path = path + filename[:-4] + ".bin"
    data = decode_binary(file_path, data_type=np.float32)
    def onpick(event):
        ind = event.ind[0]
        plot_window(ax_display, plot_display, time_array[ind]//5000, 1000, data, center=0.5)
        fig_display.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)


def get_position(target_tuple,infos):
    x_position = y_position = None
    # Iterate through the nested list to find the target tuple
    for x, sublist in enumerate(infos):
        for y, inner_list in enumerate(sublist):
            if target_tuple == inner_list:
                x_position, y_position = x, y
                break
    return x_position, y_position


def update_infofile(i,j,path):
    with open('/Users/mp274748/Documents/data_arg/pathlist.json', 'r') as json_file:
        b_list = json.load(json_file)
    path_list = b_list[0]
    infos = b_list[1]
    target_tuple = [i,j]
    x_position, y_position = get_position(target_tuple, infos)
    if y_position != None:
        path_list[x_position][y_position] = path
    else:
        infos[i].append([i,j])
        path_list[i].append(path)
    save_non_homog_list(path_list, infos, '/Users/mp274748/Documents/data_arg/pathlist.json')




def get_path(i=11, j=12):
    target_tuple = [i, j]
    with open('/Users/mp274748/Documents/data_arg/pathlist.json', 'r') as json_file:
        b_list = json.load(json_file)
    path_list = b_list[0]
    infos = b_list[1]
    x_position, y_position = get_position(target_tuple, infos)
    path = path_list[x_position][y_position]
    try:
        dictionary = load_dict(path + "dictionary.json")
    except FileNotFoundError:
        dictionary = {}
        print('no dict at this path', i,j)
    try:
        duration = dictionary["duration"]
    except KeyError:
        try:
            duration = get_duration(path, dictionary)
        except KeyError:
            return path, dictionary
    print('duration of the run :', str(datetime.timedelta(seconds=duration /4 / 5000)))
    return path, dictionary


def stab_amp(dictionary, peaks):
    try:
        stabparam, meanvalue = dictionary["stabilisation"]
        amp_stab = function_stabilize(peaks[:, 2], peaks[:, 3], stabparam, meanvalue)
    except KeyError:
        amp_stab = peaks[:, 2]
    return amp_stab


def get_pulses(dictionary, keys=['Amplitude_filtered'], type='heat'):
    dictionary_pulses_info = np.array(
        ['Trigger_position', 'Amplitude_raw', 'Amplitude_filtered', 'Baseline', 'Baseline_rms',
         'Correlation', 'TV', 'TVL', 'TVR', 'fitted_amplitude', 'Intercept', 'Rise_time',
         'Decay_time', 'Delay_amplitude', 'Mean_time', 'Surface_raw', 'Surface_filtered',
         'Chi_square', 'Controlled_fitted_amplitude', 'Controlled_intercept'])
    path = dictionary['path']
    print(path)

    try:
        filename = dictionary['filename_' + type]
    except KeyError:
        filename = dictionary['filename']
    try :
        peaks = ntd_array(path + filename)
    except FileNotFoundError:
        peaks = ntd_array('/'.join(path.split("/")[:-2]) + '/' + filename)
    dictionary_pulses = {}
    if any(key in keys for key in ['amp_stab', 'Sm', 'Energy']):
        amp_stab = stab_amp(dictionary, peaks)
        dictionary_pulses['amp_stab'] = amp_stab
    if 'Sm' in keys:
        dictionary_pulses['Sm'] = peaks[:, 9] / amp_stab
    if 'Energy' in keys:
        if type == 'heat':
            calib_function = np.poly1d(dictionary['heat_calib'])
        elif type == 'light' or type == 'trigheat':
            calib_function = np.poly1d([dictionary['light_calib'],0])
        else:
            calib_function = np.poly1d([1,0])
        dictionary_pulses['Energy'] = calib_function(amp_stab)
    for key in keys:
        try:
            i = int(np.argwhere(dictionary_pulses_info == key))
            dictionary_pulses[key] = peaks[:, i]
        except TypeError:
            pass
    return get_values(dictionary_pulses, keys)

j_run97 = [1,2,3,4,5,6,7,8,11,12]
i_s = [1, 2, 3, 4, 5, 6, 7,8,9,10,11]
j_s = [[1], [1], [1], [j for j in range(1, 9)], [0, 1], [54, 64, 4, 5, 6], [0, 1],[0,1,2,3],j_run97,j_run97,j_run97]

if __name__ == '__main__':
    infos = [[[i,j] for j in j_s[k]]for k,i in enumerate(i_s)]
    tempory = [[get_path(i,j)[0] for j in j_s[k]]for k,i in enumerate(i_s)]
    save_non_homog_list(tempory,infos,'/Users/mp274748/Documents/data_arg/pathlist.json')