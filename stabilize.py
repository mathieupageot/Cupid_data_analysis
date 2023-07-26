import numpy as np
import matplotlib.pyplot as plt

import dictionary_handler
from utility import lasso_selection
import get_data
from matplotlib.widgets import Button


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def function_stabilize(amp, baseline, stab_param, mean_value):
    p = np.poly1d(stab_param)
    return amp / (p(baseline)) * mean_value


def stabilize(amp,Sm, baseline,path):
    fig,axs = plt.subplots(2)
    fig2, axs2 = plt.subplots()
    amp_stab = []
    plt.subplots_adjust(bottom=0.25)
    button_ax = plt.axes([0.8, 0.05, 0.1, 0.05])
    button = Button(button_ax, 'Save')
    z = [0,0]
    meanvalue = 0
    pts=axs2.scatter(amp,Sm,s=0.1)
    def accept(event):

        if event.key == "enter":
            def accept2(event):
                nonlocal z
                nonlocal meanvalue
                nonlocal amp_stab
                if event.key == "enter":
                    selector2.disconnect()
                    axs[0].scatter(baseline, amp, s=0.3)
                    sel_baseline2 = selector2.xys[selector2.ind][:,0]
                    sel_amp2 = selector2.xys[selector2.ind][:,1]
                    z = np.polyfit(sel_baseline2, sel_amp2, 1).tolist()
                    p = np.poly1d(z)
                    x = np.linspace(np.min(baseline), np.max(baseline), 1000)
                    y = p(x)
                    axs[0].plot(x, y, c='r')
                    meanvalue = np.mean(sel_amp2)
                    amp_stab = function_stabilize(amp, baseline, z, meanvalue)
                    axs[1].scatter(baseline, amp_stab, s=0.1)
                    axs[1].set_ylabel('Heat Amplitude Stabilized')
                    axs[1].set_xlabel('Heat Baseline Stabilized')



                    fig.canvas.draw()

            selector.disconnect()
            sel_baseline = baseline[selector.ind]
            sel_amp = amp[selector.ind]
            pts2=axs[0].scatter(sel_baseline,sel_amp, s=0.3)
            fig.canvas.mpl_connect("key_press_event", accept2)
            selector2 = lasso_selection.SelectFromCollection(axs[0], pts2)
            fig2.canvas.draw()
            fig.canvas.draw()
    def save_stab(_):
        dictionary_handler.update_dict(path + "dictionary.json", {"stabilisation": [z, meanvalue]})
        print('Stabilized data saved, slope: ', z[0])
    axs[0].set_ylabel('Heat Amplitude Raw')
    axs[0].set_xlabel('Heat Baseline Raw')
    fig2.canvas.mpl_connect("key_press_event", accept)
    selector = lasso_selection.SelectFromCollection(axs2, pts)
    button.on_clicked(save_stab)
    plt.show()
def get_amp_stabilize(dictionary,peaks):
    try:
        stabparam, meanvalue = dictionary["stabilisation"]
        amp = function_stabilize(peaks[:, 2], peaks[:, 3], stabparam, meanvalue)
    except KeyError:
        amp = peaks[:, 2]
        print("data is stabilized")
    return amp
if __name__ == "__main__":
    path, dictionary = get_data.get_path()
    filename = dictionary['filename']
    peaks = get_data.ntd_array(path + filename)
    correlation = peaks[:, 5]
    amp = peaks[:, 2]
    Sm = peaks[:, 9] / amp
    baseline = peaks[:, 3]
    stabilize(amp, Sm, baseline, path)