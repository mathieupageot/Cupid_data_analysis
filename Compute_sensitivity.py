import matplotlib.pyplot as plt
import numpy as np

import get_data
import dictionary_handler

ms=100
def compute_sensitivity(calib, gain, type='heat'):
    if type == 'heat':
        unit_converter = 1e9
    else :
        unit_converter = 1e6
    return unit_converter/calib/gain


def save_sensitivity(i, j, type='heat'):
    path, dictionary = get_data.get_path(i,j)
    gain, calib, calib_error = dictionary_handler.get_values(dictionary, ['gain', type+'_calib', type+'_calib_error'])
    try:
        calib = calib[0]
        calib_error = calib_error[0]
    except TypeError:
        pass
    sensitivity = compute_sensitivity(calib, gain, type=type)
    sensitivity_error = sensitivity*calib_error/calib
    dictionary_handler.update_dict(path + 'dictionary.json', {type+'_sensitivity': sensitivity})
    dictionary_handler.update_dict(path + 'dictionary.json', {type + '_sensitivity_error': sensitivity_error})


def compute_FWHM(calib, sigma):
    return calib * sigma * 2.3548


def save_FWHM(i, j, type='heat'):
    path, dictionary = get_data.get_path(i, j)
    sigma, calib, sigma_error, calib_error = dictionary_handler.get_values(dictionary, ['sigma_baseline', type + '_calib',
                                                                                        "inc_sigma_baseline",
                                                                                        type + '_calib_error'])
    print(sigma, calib)
    try :
        calib = calib[0]
        calib_error = calib_error[0]
    except TypeError:
        pass
    FWHM = compute_FWHM(calib, sigma)
    FWHM_error = 2.3548 * np.sqrt((sigma*calib_error)**2+(sigma_error*calib)**2)
    dictionary_handler.update_dict(path + 'dictionary.json', {type + '_FWHM': FWHM})
    dictionary_handler.update_dict(path + 'dictionary.json', {type + '_FWHM_error': FWHM_error})


def create_table(js, save_path,i):
    labels = ['channel', 'FWHM', 'FWHM_error', 'sensitivity', 'sensitivity_error', 'Rise_time_at_200keV']
    table = np.zeros((len(js),6))
    for index,j in enumerate(js):
        path, dictionary = get_data.get_path(i,j)
        if j > 2:
            table[index,:-1] = dictionary_handler.get_values(dictionary, ['channel','heat_FWHM', 'heat_FWHM_error',
                                                              'heat_sensitivity', 'heat_sensitivity_error'])
            risetime_func = np.poly1d(dictionary['rise_time_fit'])
            table[index, -1] = risetime_func(3000)
        else :
            table[index, :-1] = dictionary_handler.get_values(dictionary, ['channel', 'light_FWHM', 'light_FWHM_error',
                                                                           'light_sensitivity',
                                                                           'light_sensitivity_error'])
            risetime_func = np.poly1d(dictionary['rise_time_fit'])
            table[index, -1] = risetime_func(200)
    with open(save_path, 'w') as file:
        # Write strings on the first line
        strings_line = ",".join(labels)
        file.write(strings_line + "\n")

        # Write numbers on the rest of the lines
        np.savetxt(file, table, fmt='%.4e,')


def plot_res_vs_sensitivity(i,js,ax):
    glue = ['UV620','UV645']
    Resistances_620 = []
    Sensitivities_620 = []
    Resistances_645 = []
    Sensitivities_645 = []
    j_620 = [5,6,11,12]
    for j in js:
        if j in j_620:
            path, dictionary = get_data.get_path(i,j)
            Resistances_620.append(dictionary['Resistance'])
            Sensitivities_620.append(dictionary['heat_sensitivity'])
        else:
            path, dictionary = get_data.get_path(i, j)
            Resistances_645.append(dictionary['Resistance'])
            Sensitivities_645.append(dictionary['heat_sensitivity'])
    ax.scatter(Resistances_645, Sensitivities_645, label = 'UV645',s=ms)
    ax.scatter(Resistances_620, Sensitivities_620, label='UV620',s=ms)
    ax.set_xlabel('Resistance in M$\Omega$')
    ax.set_ylabel('Sensitivity in nV/keV')



if __name__ == '__main__':
    js = np.array([3,4,5,6,7,8,11,12])
    '''for j in js:
        path, dict=get_data.get_path(11,j)
        dictionary_handler.update_dict(path+'dictionary.json', {'gain': 910})
    for j in js[:3]:
        save_sensitivity(11, j, type='light')
        save_FWHM(11, j, type='light')
    for j in js[2:]:
        save_sensitivity(11, j, type='heat')
        save_FWHM(11, j, type='heat')
    create_table(js, '/Users/mp274748/Documents/data_arg/RUN97/meas4/infofile.txt',11)'''
    save_plot = 1
    if save_plot == 1:
        import matplotlib.pylab as pylab

        params = {'legend.fontsize': 20.,
                  'figure.figsize': (15, 5),
                  'axes.labelsize': 20.,
                  'axes.titlesize': 20.,
                  'xtick.labelsize': 20.,
                  'ytick.labelsize': 20.}
        pylab.rcParams.update(params)
    fig, ax = plt.subplots()
    plot_res_vs_sensitivity(9, js, ax)
    R = [2.4,2.4,4.3,3.7,0.9,2.2,0.54,4.8,2.7,5.3,3]
    S = [19,14,35,22,5,11,3,32,8,40,29]
    ax.scatter(R,S,label='CROSS crystals', marker= '+',s=ms)
    R_ara = [4.1,9.9,11.7,4.1]
    S_ara = [29,37,33,4]
    ax.scatter(R_ara, S_ara, label='RUN93 Araldite', marker='*',s=ms)
    R_uv = [6.5,5.4 ,4.7,5.4]
    S_uv = [7,4,19, 7]
    ax.scatter(R_uv, S_uv, label='RUN93 UV645', marker='v',s=ms)
    plt.legend()
    plt.show()


