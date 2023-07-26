import matplotlib.pyplot as plt

if __name__ == "__main__":
    from stabilize import stabilize
    from cuts import get_cut_para, cut_function
    from peakdetect import get_calibration
    import get_data
    import numpy as np
    from risetime import get_risetime
    #path = input('path to the dictionary:')
    #get_data.update_infofile(path,12,3)
    path, dictionary = get_data.get_path(11, 3)
    Amplitude_not_stab, fitted_amplitude, Baseline = get_data.get_pulses(dictionary,
                                                                         ['Amplitude_filtered', 'fitted_amplitude',
                                                                          'Baseline'])
    Sm = fitted_amplitude / Amplitude_not_stab
    stabilize(Amplitude_not_stab, Sm, Baseline, path)
    plt.show()

    correlation, amplitude, rise_time = get_data.get_pulses(dictionary, ['Correlation', 'amp_stab',
                                                                         'Rise_time'])
    amp_lim = 0.5
    para_correlation = np.array([-0.001, -1, 0.95, 1.e-5])
    get_cut_para(amplitude[amplitude < amp_lim], para_correlation, path, correlation[amplitude < amp_lim])
    plt.show()

    para_correlation = dictionary['correlation']
    selection = correlation > cut_function(amplitude, *para_correlation)
    selected_amp = amplitude[selection]
    max_bin = 1500
    change_peak_safe = False
    peak_safe = np.array([0.4050])
    selected_amp = selected_amp[selected_amp < peak_safe * 1.1]
    energies = np.array([352, 609, 1120, 1764])
    get_calibration(selected_amp, energies, dictionary, peak_safe=peak_safe, energy_safe=[5000])
    plt.show()

    fig_rt, ax_rt = plt.subplots()
    get_risetime(selected_amp, rise_time[selection], ax_rt, len(selected_amp) // 30)
    plt.show()
