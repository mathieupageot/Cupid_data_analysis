import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def sigma_function(x, a, b, c, d):
    return a * x ** b + c + d*x

def gaussians_from_scatter(x_scat,y_scat,events_per_interval=500):
    # Calculate the x-interval size
    sorted_indices = np.argsort(x_scat)
    sorted_x = x_scat[sorted_indices]
    sorted_y = y_scat[sorted_indices]

    # Calculate the number of intervals based on the total number of events and the desired number of events per interval
    num_intervals = len(x_scat) // events_per_interval
    # Split the sorted data into intervals
    x_intervals = np.array_split(sorted_x, num_intervals)
    y_intervals = np.array_split(sorted_y, num_intervals)
    popts = []
    pcovs = []
    interval_centers = []
    fig2,ax2 = plt.subplots(3,3)
    # Iterate over partitions
    for i,(x_interval, y_interval) in enumerate(zip(x_intervals,y_intervals)):
        center = np.mean(x_interval)
        interval_centers.append(center)
        hist, bins = np.histogram(y_interval,bins=events_per_interval*3)
        bins_center = (bins[:-1] + bins[1:]) / 2
        try :
            popt, pcov = curve_fit(gaussian, bins_center, hist, [hist.max(), y_interval.mean(), np.std(y_interval)])
        except RuntimeError:
            popt, pcov = np.zeros(3), np.zeros((3,3))
        if i<9:
            x_plot=np.linspace(popt[1]-5*popt[2],popt[1]+5*popt[2],1000)
            ax2[i//3,i%3].plot(bins_center,hist,linewidth=.5, ds='steps-mid')
            ax2[i // 3, i % 3].plot(x_plot,gaussian(x_plot,*popt))
            ax2[i // 3, i % 3].set_xlim(popt[1]-5*popt[2],popt[1]+5*popt[2])



        popts.append(popt)
        pcovs.append(pcov)
    return np.array(popts), np.array(pcovs), interval_centers

def plot_scatter_with_gauss(x_scat,y_scat,ax_scatter,num_partitions = 5):
    mean_y_scat = np.mean(y_scat)
    std_y_scat = np.std(y_scat)
    sel_y = np.logical_and(y_scat<mean_y_scat+10*std_y_scat,y_scat>mean_y_scat-10*std_y_scat)
    x_scat, y_scat = x_scat[sel_y],y_scat[sel_y]
    ax_scatter.scatter(x_scat, y_scat, s=0.1,c='b')
    popts, pcovs, interval_centers = gaussians_from_scatter(x_scat,y_scat,num_partitions)
    sigmas = popts[:,2]
    sigmas = np.abs(sigmas)
    y0s = popts[:,1]
    ax_scatter.scatter(interval_centers,y0s,c='g',s=10, label='mean value')
    ax_scatter.scatter(interval_centers, y0s+5*sigmas, c='r',s=10)
    ax_scatter.scatter(interval_centers, y0s-5*sigmas, c='r',s=10)
    x_fit, y_fit = interval_centers,y0s+5*sigmas
    y_fit_bis = y0s-5*sigmas
    p_guess = [x_fit[0]*y_fit[0],-1,y0s[0],0]

    x_plot = np.linspace(x_scat.min(),x_scat.max(),3000)
    try :
        popt_sigma, pcov_sigma = curve_fit(sigma_function, x_fit, y_fit, p_guess)
    except RuntimeError:
        popt_sigma = p_guess
    p_guess_bis = [-popt_sigma[0], popt_sigma[1], 2 * y0s.mean() - popt_sigma[2],0]
    try :
        popt_sigma_bis,_= curve_fit(sigma_function, x_fit, y_fit_bis, p_guess_bis)
    except RuntimeError:
        popt_sigma_bis = p_guess_bis
    ax_scatter.plot(x_plot,sigma_function(x_plot,*popt_sigma),c='r',label = '5$\sigma$ interval')
    ax_scatter.plot(x_plot, sigma_function(x_plot, *popt_sigma_bis), c='r')
    ax_scatter.text(0.5, 0.60,"mean Rise time = {:.2e} ms".format(y0s.mean()*1000),bbox=dict(facecolor='white', edgecolor='white'), fontsize=12, color='black', transform=ax_scatter.transAxes,verticalalignment='top')
    lin_para,_ = curve_fit(lambda x,a,b : a * x +b ,interval_centers, y0s,[0,y0s.mean()])
    print(popt_sigma)
    print(lin_para)
    ax_scatter.plot([0,300],lin_para[0]*np.array([0,300])+lin_para[1])
    ax_scatter.set_xlabel("Energy in keV")
    ax_scatter.set_ylabel("Rise time in s")
    ax_scatter.set_title('Energy vs Rise time for light detector channel 2')
    fig = ax_scatter.get_figure()
    fig.legend()

if __name__ == "__main__":
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 20.,
              'figure.figsize': (15, 5),
              'axes.labelsize': 20.,
              'axes.titlesize': 20.,
              'xtick.labelsize': 20.,
              'ytick.labelsize': 20.}
    pylab.rcParams.update(params)
    import get_data
    path, dictionary = get_data.get_path()
    print(dictionary)
    filename, filename_light, filename_trigheat = get_data.get_values(dictionary,
                                                                                     ["filename", "filename_light",
                                                                                      "filename_trigheat" ])
    ampl, risetl = get_data.import_light(path, filename_light)
    coeff_light, error_coeff_light = dictionary["light_calib"]
    ampl_fit = ampl * coeff_light
    sel_rt = risetl<1
    ampl_fit, risetl = ampl_fit[sel_rt], risetl[sel_rt]
    fig, ax = plt.subplots()
    Energy_max = 150
    plot_scatter_with_gauss(ampl_fit[ampl_fit<Energy_max],risetl[ampl_fit<Energy_max],ax,len(ampl_fit[ampl_fit<Energy_max])//30)
    #ax.scatter(ampl_fit,risetl,s=0.1)
    #ax.hist(ampl_fit,1000)
    plt.show()

