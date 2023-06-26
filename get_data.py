import numpy as np
import datetime
import json


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

def get_path(i=7, j=1, peak=False):
    # input("file?")
    j = int(j)
    filenum = i
    filebis = j

    if i == 3:  # measurement 0003
        path = '/Users/mp274748/Documents/data_arg/second_set/'
        filename = '20211125_00h43.BINLMO21.2.ntp'
        filename_light = '20211125_00h43.BINLD21.2.ntp'
        filename_trigheat = '20211125_00h43.BINLD21.2_trigheat.ntp'
    if i == 2:
        path = '/Users/mp274748/Documents/data_arg/first_set2/'
        filename = '20180709_23h07.BINLMO.ntp'
        filename_light = '20180709_23h07.BINLD.ntp'
        filename_trigheat = '20180709_23h07.BINLD_trigheat0.ntp'
    if i == 1:
        path = '/Users/mp274748/Documents/data_arg/first_set/'
        filename = '20180709_23h07.BINLMO.ntp'
        filename_light = '20180709_23h07.BINLD.ntp'
        filename_trigheat = '20180709_23h07.BINLD_trigheat.ntp'
    if i == 4:
        filename = '20230308_16h15.BIN'
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
        else:
            name = ''
        path += name + '/'
        filename += name + '.ntp'
        filename_light = 0
        filename_trigheat = 0
    # data from edi
    if filenum == 5:
        filename = 'LoggedData_2021_05_05_17_40_40.NTD_'
        path = '/Users/mp274748/Documents/data_edi/'
        if filebis == 0:
            name = 'RAW'
        if filebis == 1:
            name = 'DIF'
        path += name + '/'
        filename += name + '.ntp'
        filename_light = 0
        filename_trigheat = 0
    # data form RUN96 measurement 2
    if filenum == 6:
        if filebis == 54:
            filename_light = '000004_20230606T192243_005_trig4.ntp'
            path = '/Users/mp274748/Documents/data_arg/RUN96/Measurement2/trig4/'
        elif filebis == 64:
            filename_light = '000004_20230606T192243_006_trig4.ntp'
            path = '/Users/mp274748/Documents/data_arg/RUN96/Measurement2/trig4/'
        else:
            filename_light = '000004_20230606T192243_00' + str(filebis) + '_001.bin.ntp'
            path = '/Users/mp274748/Documents/data_arg/RUN96/Measurement2/channel' + str(filebis) + '/'
        filename = 0
        filename_trigheat = 0
    # data form RUN96 measurement 6
    if filenum == 7:
        path = '/Users/mp274748/Documents/data_arg/RUN96/Measurement6/'
        if filebis == 0:
            path += 'LMO18/'
            filename = '000015_20230609T235012_008.ntp'
            filename_light = '000015_20230609T235012_001_bis.ntp'
            filename_trigheat = '000015_20230609T235012_001_trig.ntp'
        elif filebis == 1:
            path += 'LMO26/'
            filename = '000015_20230609T235012_007.ntp'
            filename_light = '000015_20230609T235012_002.ntp'
            filename_trigheat = '000015_20230609T235012_002_trig.ntp'
    try:
        print('file: ' + filename)
    except TypeError:
        try:
            print('file: ' + filename_light)
        except TypeError:
            print('No file found')

    dict = {"path": path, "filename": filename, "filename_light": filename_light,
            "filename_trigheat": filename_trigheat}
    try :
        for key, value in zip(dict.keys(), dict.values()):
            update_dict(path + "dictionary.json", key, value)
    except FileNotFoundError:
        with open(path + "dictionary.json", 'w') as json_file:
            json.dump(dict, json_file)
    dictionary = load_dict(path + "dictionary.json")
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
    print("acquisition duration(h:m:s):",
          str(datetime.timedelta(hours=np.round((peaks[-1, 0] - peaks[0, 0]) / 5000 / 60 / 60, 2))))
    return E, amp_stab, correlation, TV, riset, decayt, Sm, time


def import_light(path, filename_light):
    peaksl = ntd_array(path + filename_light)
    correlation = peaksl[:, 5]
    sel_corr = correlation>0.95
    ampl = peaksl[sel_corr, 2]
    riset = peaksl[sel_corr, 11]
    return ampl,riset
