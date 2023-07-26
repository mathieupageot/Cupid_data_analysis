import get_data
import numpy as np


def find_closest_indices(N, M):
    indices = np.searchsorted(N, M)
    closest_indices = np.where((indices == 0) | (indices == len(N)) | (np.abs(N[indices] - M) < np.abs(N[indices - 1] - M)), indices, indices - 1)
    return closest_indices


def plot_delay_vs_time(Trigger_position_1, Trigger_position_2, delay_range, ax):
    delay = Trigger_position_1 - Trigger_position_2
    selection_time = np.logical_and(delay < delay_range, delay > -delay_range)
    ax.plot(energy[selection_time], delay[selection_time], marker='o', linestyle='', ms=0.1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    '''path_RAW, dictionary_RAW = get_data.get_path(5, 0)
    Trigger_position_RAW = get_data.get_pulses(dictionary_RAW, ['Trigger_position'])[0]
    path_DIF, dictionary_DIF = get_data.get_path(5, 1)
    Trigger_position_DIF = get_data.get_pulses(dictionary_DIF, ['Trigger_position'])[0]'''
    path, dictionary = get_data.get_path(8, 3)
    Trigger_position_heat, energy = get_data.get_pulses(dictionary, ['Mean_time', 'Energy'])
    Trigger_position_light = get_data.get_pulses(dictionary, ['Mean_time'], type='trigheat')[0]
    print(Trigger_position_heat[0:10])
    print(Trigger_position_light[0:10])
    #closest_indices = find_closest_indices(Trigger_position_heat, Trigger_position_light)
    fig_delay, ax_delay = plt.subplots()
    plot_delay_vs_time(Trigger_position_heat, Trigger_position_light, 100e10, ax_delay)
    plt.show()
