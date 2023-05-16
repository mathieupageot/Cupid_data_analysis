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
Sm = peaks[:, 9] / amp
E = p(amp)


def gauss(x, x0, sigma, A):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def lin(x, a, b, c):
    return a * x ** b + c


sigmas = []
mus = []
En = np.arange(0, 2000, 50)
centerE = (En[:-1] + En[1:]) / 2
for i in range(len(En) - 1):
    dE = np.logical_and(E > En[i], E < En[i + 1])
    selSm = Sm[dE]
    n, bins = np.histogram(selSm, 100)
    center = (bins[:-1] + bins[1:]) / 2

    param, _ = curve_fit(gauss, center, n, [1, 0.05, 20])
    sigmas.append(3 * np.abs(param[1]))
    mus.append(param[0])
    x = np.linspace(center.min(), center.max(), 1000)
    if i == 9:
        fig,ax=plt.subplots()
        ax.plot(center,n)
        ax.plot(x,gauss(x,*param))
sigmas = np.array(sigmas)
mus=np.array(mus)
fig2, ax2 = plt.subplots()
ax2.scatter(E, Sm, s=0.1)
p0_up = [1,-1,1.04]
p0_down = [-1,-1,0.97]
popt_up, _ = curve_fit(lin, centerE, sigmas + mus,p0_up)
popt_down, _ = curve_fit(lin, centerE, -sigmas + mus,p0_down)

x = np.linspace(centerE.min(), centerE.max(), 1000)
ax2.plot(x,lin(x,*popt_up),c='r')
ax2.plot(x,lin(x,*popt_down),c='g')
ax2.scatter(centerE, sigmas + mus)
ax2.scatter(centerE, -sigmas + mus)
plt.show()
