import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import get_data
path, filename, filename_light, filename_trigheat, filenum = get_data.get_path()
peaks = get_data.ntd_array(path + filename)
if filenum == 2:
    z = [0.54076489, 6.23910689]
if filenum == 3:
    z = [2.47881842, -10.57679343]
else:
    z = [1, 0]
p = np.poly1d(z)
amp = np.load(path + 'amp_stab.npy')
Sm = peaks[:, 9]/amp
E=p(amp)
def gauss(x,x0,sigma,A):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

sigmas=[]

En=np.arange(0,6000,300)
centerE = (En[:-1] + En[1:]) / 2
axs=[]
for i in range(len(En)-1):

    dE=np.logical_and(E>En[i],E<En[i+1])
    selSm = Sm[dE]
    n, bins = np.histogram(selSm, 200) #int((selSm.max()-selSm.min())/0.001)
    center = (bins[:-1] + bins[1:]) / 2

    param,_ = curve_fit(gauss,center,n,[1,0.05,20])
    sigmas.append(3*param[1])
    x=np.linspace(center.min(),center.max(),1000)
    # fig,ax = plt.subplots()
    # axs.append(ax)
    #axs[-1].plot(center,n)
    #axs[-1].plot(x,gauss(x,*param))

plt.scatter(E,Sm,s=0.1)
sigmas=np.array(sigmas)
plt.scatter(centerE,sigmas+1)
plt.scatter(centerE,-sigmas+1)
plt.show()
