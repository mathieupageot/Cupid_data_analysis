import numpy as np
def ntd_array(path):
    A=np.loadtxt(path)
    return A

if __name__=="__main__":
    arr=ntd_array(r'/Users/mp274748/Documents/20180709_23h07.BINLD.ntp')
    print(np.shape(arr))