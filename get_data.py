import numpy as np
import datetime
import json
def find_local_maxima(data):
    local_maxima = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            local_maxima.append(data[i])
    local_maxima.append(data[-1])
    return local_maxima
def get_duration(i,j):
    path, dictionary = get_path(i, j)
    filename = dictionary["filename"]
    try:
        peaks = ntd_array(path + filename)
    except TypeError:
        filename = dictionary["filename_light"]
        peaks = ntd_array(path + filename)
    duration = np.sum(find_local_maxima(peaks[:, 0]))
    update_dict(path + "dictionary.json", "duration", duration)
    return duration
def ntd_array(path):
    print(path)
    A = np.loadtxt(path, usecols=(n for n in range(20)))
    return A


def load_dict(dict_path):
    with open(dict_path, 'r') as json_file:
        dict = json.load(json_file)
    return dict


def update_dict(dict_path, key, value):
    with open(dict_path, 'r') as json_file:
        data = json.load(json_file)
    try:
        value = value.tolist()
    except:
        pass
    data[key] = value
    with open(dict_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
def get_values(dictionary,key_list):
    value_set = []
    for key in key_list:
        if key in dictionary:
            print(dictionary[key])
            value_set.append(dictionary[key])
    return value_set


def get_path(i=7, j=1):
    # input("file?")
    j = int(j)
    filenum = i
    filebis = j

    if i == 3:  # measurement 0003
        path = '/Users/mp274748/Documents/data_arg/second_set/'
    if i == 2:
        path = '/Users/mp274748/Documents/data_arg/first_set2/'
    if i == 1:
        path = '/Users/mp274748/Documents/data_arg/first_set/'
    if i == 4:
        path = '/Users/mp274748/Documents/data_arg/RUN93/'
        if j == 1:
            name = '60b_ara1'
        elif j == 2:
            name = '60b_ara2'
        elif j == 3:
            name = '61b_ara1'
        elif j == 4:
            name = '61b_ara2'
        elif j == 5:
            name = '60b_UV1'
        elif j == 6:
            name = '60b_UV2'
        elif j == 7:
            name = '61b_UV1'
        elif j == 8:
            name = '61b_UV2'
        path += name + '/'
    # data from edi
    if filenum == 5:
        path = '/Users/mp274748/Documents/data_edi/'
        if filebis == 0:
            name = 'RAW'
        if filebis == 1:
            name = 'DIF'
        path += name + '/'
        filename_light = 0
        filename_trigheat = 0
    # data form RUN96 measurement 2
    if filenum == 6:
        if filebis == 54:
            path = '/Users/mp274748/Documents/data_arg/RUN96/Measurement2/5_trig4/'
        elif filebis == 64:
            path = '/Users/mp274748/Documents/data_arg/RUN96/Measurement2/6_trig4/'
        else:
            path = '/Users/mp274748/Documents/data_arg/RUN96/Measurement2/channel' + str(filebis) + '/'
    # data form RUN96 measurement 6
    if filenum == 7:
        path = '/Users/mp274748/Documents/data_arg/RUN96/Measurement6/'
        if filebis == 0:
            path += 'LMO18/'
        elif filebis == 1:
            path += 'LMO26/'
    dictionary = load_dict(path + "dictionary.json")
    try:
        duration = dictionary["duration"]
    except ValueError:
        duration = get_duration(i,j)
    print('duration of the run :', str(datetime.timedelta(seconds=duration/5000)))
    return path, dictionary


def get_heat(path, filename, p, dict):
    from stabilize import function_stabilize
    peaks = ntd_array(path + filename)
    try:
        stabparam, meanvalue = dict["stabilisation"]
        amp_stab = function_stabilize(peaks[:, 2], peaks[:, 3], stabparam, meanvalue)
    except:
        amp_stab = np.load(path + 'amp_stab.npy')
    E = p(amp_stab)
    correlation = peaks[:, 5]
    TV = peaks[:, 8]
    riset = peaks[:, 11]
    decayt = peaks[:, 12]
    Sm = peaks[:, 9] / amp_stab
    time = peaks[:, 0]
    return E, amp_stab, correlation, TV, riset, decayt, Sm, time


def import_light(path, filename_light):
    peaksl = ntd_array(path + filename_light)
    correlation = peaksl[:, 5]
    sel_corr = correlation>0.95
    ampl = peaksl[sel_corr, 2]
    riset = peaksl[sel_corr, 11]
    return ampl,riset

i_s = [1,2,3,4,5,6,7]
j_s = [[1],[1],[1],[j for j in range(1,9)],[0,1],[54,64,4,5,6],[0,1]]