import numpy as np
import datetime
def ntd_array(path):
    A=np.loadtxt(path)
    return A
def get_path(i=3,j=2,peak=False):
    j=1#input("file?")
    j=int(j)
    filenum=i
    filebis=j

    if i == 3: #measurement 0003
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
        else :
            name = ''
    if filenum == 2:
        data_E = np.array([352, 609, 295, 242])
        data_amp = np.array([1114, 640, 537, 433])
    if filenum == 3:
        data_E = np.array([352, 609, 768, 1120, 1238, 1764])
        data_amp = np.array([145, 319, 502, 246, 460, 714])
    if filenum == 4:
        if filebis == 1:
            data_E = np.array([510.77, 583.2, 609.312, 911.204, 2614.533])
            data_amp = np.array([254, 298, 313, 472, 1362])
        if filebis == 2:
            data_E = np.array([510.77, 583.2, 609.312, 911.204, 2614.533])
            data_amp = np.array([274, 313, 335, 572, 1822])
        if filebis == 5:
            data_E = np.array([583.2, 911.204, 968.971, 2614.533])
            data_amp = np.array([147, 112, 154, 340])
        if filebis == 6:
            data_E = np.array([583.2, 911.204, 968.971, 2614.533])
            data_amp = np.array([63, 81, 85, 185])
        if filebis == 3:
            data_E = np.array([510.77, 583.191, 911.204, 968.971, 2614.533])
            data_amp = np.array([265, 323, 544, 586, 1692])
        if filebis == 4:
            data_E = np.array([583.191, 609.312, 911.204, 968.971, 2614.533])
            data_amp = np.array([61.4, 63, 84.6, 88.2, 207])
        if filebis == 7:
            data_E = np.array([510.77, 609.312, 583.191, 911.204, 968.971, 2614.533])
            data_amp = np.array([164, 183, 198, 318, 350, 1006])
        if filebis == 8:
            data_E = np.array([583.191, 911.204, 968.971, 2614.533])
            data_amp = np.array([93, 141, 151, 384])
        path += name + '/'
        filename += name + '.ntp'
        filename_light = 0
        filename_trigheat = 0
    if filenum == 5:
        filename = 'LoggedData_2021_05_05_17_40_40.NTD_'
        path = '/Users/mp274748/Documents/data_edi/'
        if filebis == 0:
            name = 'RAW'
            data_E = [0, 0.1, 0.128]
            data_amp = [0, 0.1, 0.128]
        if filebis == 1:
            name = 'DIF'
            data_E = [0, 0.1, 0.128]
            data_amp = [0, 0.1, 0.128]
        path += name + '/'
        filename += name + '.ntp'
        filename_light = 0
        filename_trigheat = 0


    print('file: '+filename)
    if peak:
        return path,filename,filename_light,filename_trigheat,np.sort(data_E),np.sort(data_amp)
    else:
        z = np.polyfit(data_amp, data_E, 1)
        p = np.poly1d(z)
        return path, filename, filename_light, filename_trigheat,p
def get_heat(path,filename,p):
    peaks = ntd_array(path + filename)
    amp_stab = np.load(path + 'amp_stab.npy')
    E = p(amp_stab)
    correlation = peaks[:, 5]
    TV = peaks[:, 8]
    riset = peaks[:, 11]
    decayt = peaks[:, 12]
    Sm = peaks[:, 9] / amp_stab
    time = peaks[:,0]
    print("acquisition duration(h:m:s):",str(datetime.timedelta(hours=np.round((peaks[-1,0]-peaks[0,0])/5000/60/60,2))))
    return E,amp_stab,correlation,TV,riset,decayt,Sm,time
if __name__=="__main__":
    arr=ntd_array(r'/Users/mp274748/Documents/20180709_23h07.BINLD.ntp')
    print(np.shape(arr))