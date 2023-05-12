import numpy as np
def ntd_array(path):
    A=np.loadtxt(path)
    return A
def get_path():
    i = 3#int(input("Data_set: "))
    print(i)

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



    return path,filename,filename_light,filename_trigheat,i

if __name__=="__main__":
    arr=ntd_array(r'/Users/mp274748/Documents/20180709_23h07.BINLD.ntp')
    print(np.shape(arr))