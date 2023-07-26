import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import Button
import dictionary_handler
def linear(x, a, b):
    return a * x +b


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def sigma_function(x, a, b, c):
    return a * x ** b + c

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
    #fig2,ax2 = plt.subplots(3,3)
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
        if i<-1:
            x_plot=np.linspace(popt[1]-5*popt[2],popt[1]+5*popt[2],1000)
            ax2[i//3,i%3].plot(bins_center,hist,linewidth=.5, ds='steps-mid')
            ax2[i // 3, i % 3].plot(x_plot,gaussian(x_plot,*popt))
            ax2[i // 3, i % 3].set_xlim(popt[1]-5*popt[2],popt[1]+5*popt[2])



        popts.append(popt)
        pcovs.append(pcov)
    return np.array(popts), np.array(pcovs), interval_centers

def get_risetime(x_scat, y_scat, ax_scatter, num_partitions = 5):
    mean_y_scat = np.mean(y_scat)
    std_y_scat = np.std(y_scat)
    sel_y = np.logical_and(y_scat<mean_y_scat+10*std_y_scat,y_scat>mean_y_scat-10*std_y_scat)
    x_scat, y_scat = x_scat[sel_y],y_scat[sel_y]
    scatter_data = ax_scatter.scatter(x_scat, y_scat, s=0.1,c='b')
    popts, pcovs, interval_centers = gaussians_from_scatter(x_scat,y_scat,num_partitions)
    n_min_fit = 8
    n_lin_fit = num_partitions-20
    interval_centers = np.array(interval_centers[n_min_fit:])
    sigmas = popts[n_min_fit:,2]
    sigmas = np.abs(sigmas)
    y0s = np.array(popts[n_min_fit:,1])
    y0s_error = np.sqrt(pcovs[n_min_fit:,1,1]).reshape(-1)
    ax_scatter.scatter(interval_centers,y0s,c='y',s=10, label='mean value')
    ax_scatter.scatter(interval_centers, y0s+5*sigmas, c='r',s=10)
    ax_scatter.scatter(interval_centers, y0s-5*sigmas, c='r',s=10)
    x_plot = np.linspace(x_scat.min(),x_scat.max(),3000)
    lin_para, lin_cov = curve_fit(linear, interval_centers, y0s,sigma=y0s_error)
    print(lin_para)
    y1s = y0s / linear(interval_centers, *lin_para)
    sigma1s = sigmas / linear(interval_centers, *lin_para)
    y_fit = y0s + 5 * sigma1s
    bool_sigma_fit=0
    if bool_sigma_fit == 1:
        guess = [interval_centers[0] * y_fit[0], -1, 0]
        popt_sigma, _ = curve_fit(sigma_function, interval_centers[-n_lin_fit:],
                                  y1s[-n_lin_fit:] + 5 * sigma1s[-n_lin_fit:], p0=guess)
        popt_sigma_bis = [-popt_sigma[0], popt_sigma[1], 2 - popt_sigma[2]]
        ax_scatter.plot(x_plot, sigma_function(x_plot, *popt_sigma)*linear(x_plot,*lin_para), c='r', label='5$\sigma$ interval')
        ax_scatter.plot(x_plot, sigma_function(x_plot, *popt_sigma_bis)*linear(x_plot,*lin_para), c='r')
    ax_scatter.text(0.75, 0.80, "Rise time at 300keV = {:.2e} ms".format(linear(300,*lin_para)),
                    bbox=dict(facecolor='white', edgecolor='white'), fontsize=12, color='black',
                    transform=ax_scatter.transAxes, verticalalignment='top')

    ax_scatter.plot(x_plot,linear(x_plot,*lin_para),c='y')
    ax_scatter.set_xlabel("Energy in keV")
    ax_scatter.set_ylabel("Rise time in s")
    ax_scatter.set_title("Energy vs Rise time for light detector channel {}".format(int(dictionary['channel'])))
    fig = ax_scatter.get_figure()
    fig.legend()
    def safe_calib(_):
        dictio_update = {'rise_time_fit': list(lin_para), 'rise_time_fit_error':list(np.sqrt(np.diag(lin_cov)))}
        dictionary_handler.update_dict(path + "dictionary.json",
                                       dictio_update)
        print('saved')
    save_button_ax = fig.add_axes([0.92, 0.05, 0.05, 0.1])  # Define button axes coordinates
    save_button = Button(save_button_ax, 'Save')
    save_button.on_clicked(safe_calib)
    plt.show()
    return scatter_data

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
    path, dictionary = get_data.get_path(9,2)
    ampl_fit, risetl = get_data.get_pulses(dictionary, ['Energy', 'Rise_time'],type='light')
    #ampl_fit, risetl = get_data.get_pulses(dictionary, ['Energy', 'Rise_time'], type='heat')
    sel_rt = risetl<1
    ampl_fit, risetl = ampl_fit[sel_rt], 1000* risetl[sel_rt]
    fig, ax = plt.subplots()
    Energy_max = 400
    get_risetime(ampl_fit[ampl_fit < Energy_max], risetl[ampl_fit < Energy_max], ax, len(ampl_fit[ampl_fit < Energy_max]) // 30)
    plt.show()

