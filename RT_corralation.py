import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utility.double_axis_display import make_format
import utility.file_handler as file_handler


save_plot = 1
if save_plot == 1:
    import matplotlib.pylab as pylab

    params = {'legend.fontsize': 20.,
              'figure.figsize': (15, 5),
              'axes.labelsize': 20.,
              'axes.titlesize': 20.,
              'xtick.labelsize': 20.,
              'ytick.labelsize': 20.}
    pylab.rcParams.update(params)


def linear_function(x, a, b):
    return a * x + b


def get_position(A, B):
    C = np.full_like(A,np.nan,dtype=np.float64)
    # Find the indices of elements in B that are present in A
    inB = np.isin(A, B)
    C[inB]=[np.where(B==a)[0][0] for a in A[inB]]
    return C

class changing_functions :
    def __init__(self,ref_array):
        self.ref_array = ref_array
        self.c = np.min(ref_array)
        self.d = np.max(ref_array)
    def f(self,x):
        return get_position(x,self.ref_array)
    def f_1(self,x):
        a = np.min(x)
        b = np.max(x)
        c = self.c
        d = self.d
        return np.round(c + ((x - a) * (d - c)) / (b - a)).astype(int)


filename = 'trmc2_2023-07-26.xls'
path = "/Users/mp274748/Documents/data_arg/RUN97/"
df = pd.read_excel(path + filename)
column_names = np.array(df.columns.tolist())
find_ntds = np.logical_and(['(meas.)' in column_name for column_name in column_names],
                           [('2' in column_name or '3' in column_name) for column_name in column_names])
pos_ntds = np.argwhere(find_ntds)
R_ntds = [df[name].values for name in column_names[find_ntds]]
Temperature = df['1Ed_RuO2_float_plate (conv.)'].values
popts = []
selection = np.abs(np.gradient(Temperature)) < 0.003
names = ['B645_1', 'U645_1', 'B620_1', 'U620_1', 'B645_2', 'U645_2', 'B620_2', 'U620_2']
dictionary = {}
table = []
js = np.array([3, 4, 5, 6, 7, 8, 11, 12])
fig, ax = plt.subplots()
R0s, T0s = [], []


def x1_to_x2(x):
    return 1 / np.sqrt(x)


# Function to transform x2 back to x1 values
def x2_to_x1(x):
    return 1 / x ** 2


axT = ax.twiny()
axT.format_coord = make_format(axT, ax)
axT.set_xscale('function', functions=(x1_to_x2, x2_to_x1))
for arg, Res in enumerate(R_ntds):
    filed_lines = np.logical_and(pd.notna(Res), selection)
    x = 1 / np.sqrt(Temperature[filed_lines])
    sel2 = np.logical_and(x < 6.5, x > 2.864)
    y = np.log(Res[filed_lines])
    x, y = x[sel2], y[sel2]
    Tbis = Temperature[filed_lines]
    Tbis = Tbis[sel2]

    if len(y) > 0:
        l, = ax.plot(x[::10], y[::10], ls='', marker='')
        axT.plot(Tbis[::10] * 1000, y[::10], ls='', marker='+')
        sel_fit = np.logical_and(x < 5, x > 3.)
        popt, pcov = curve_fit(linear_function, x[sel_fit], y[sel_fit])
        ax.plot(x[sel_fit], linear_function(x[sel_fit], *popt), c=l.get_color(), label=names[arg])
        p_error = np.sqrt(np.diag(pcov))
        R0 = np.exp(popt[1])
        T0 = popt[0] ** 2
        dR = R0 * p_error[1]
        dT = 2 * popt[0] * p_error[0]
        table.append([js[arg], R0, dR, T0, dT])
        T0s.append(T0)
        R0s.append(R0)
ax.set_xlabel('1/$\sqrt{T}$ (K)')
axT.set_xlabel('T (mK)')
ax.set_ylabel('log(R) ($\Omega$)')
fig.legend(framealpha=1)
T0s = np.array(T0s)
R0s = np.array((R0s))


def plot_value_on_channel(channel_array, y_values, split_arrays, labels, markers=None, y_label=''):
    if markers == None :
        markers = np.full_like(channel_array,"o",dtype=str)
    changing_function = changing_functions(np.concatenate(([channel_array[0]-1],channel_array,[channel_array[-1]+1])))
    fig, ax = plt.subplots()
    ax.set_xscale('function', functions=(changing_function.f, changing_function.f_1))
    for split_array,label,marker in zip(split_arrays,labels,markers):
        ax.scatter(channel_array[split_array], y_values[split_array],label=label,s=100,marker=marker)
    ax.set_xticks(channel_array)
    ax.set_xlabel('Channel')
    ax.set_ylabel(y_label)
    plt.legend()

labels = ['NTD name', 'R0', 'dR0', 'T0', 'dT0']
file_handler.save_array_with_label(labels, table, path+'RT_correlation.txt')
channels645 = np.array([True,True,False,False,True,True,False,False])
channels620 = np.logical_not(channels645)
channel_array = np.array([3,4,5,6,7,8,11,12])
plot_value_on_channel(channel_array, T0s, [channels645, channels620], ['UV645', 'UV620'], ['s', 'v'], '$T_0$(K)')
plot_value_on_channel(channel_array, R0s, [channels645, channels620], ['UV645', 'UV620'], ['s', 'v'], '$R_0$($\Omega$)')



def compute_reduced_sensibility(Sensibility, T0, R0, Vbol, R):
    T = T0 / np.log(R / R0)**2
    A = np.log(R / R0) / 2
    return Sensibility / (A * Vbol) *1.e-3


def compute_real_temperature(T0, R0, R):
    return T0 / np.log(R / R0) ** 2


def get_Rbol_Vbol(channel_num, bias_V, V_I_path):
    df = pd.read_excel(V_I_path)
    bias_Vs = df['Bias_V'].values
    channels = df['Name'].values
    R_bols = df['R_Bol'].values
    V_bols = df['V_Bol'].values
    working_point = np.logical_and(bias_Vs == bias_V, channels == "CH{}".format(channel_num))
    return R_bols[working_point], V_bols[working_point]


def plot_reduced_sensibility(channels, bias_Vs, T0s, R0s, Sensibilities,V_I_path):
    ethas = np.zeros_like(bias_Vs)
    R_bols = np.zeros_like(bias_Vs)
    fig, ax = plt.subplots()
    for i, (channel, bias_V, T0, R0, Sensibility) in enumerate(zip(channels, bias_Vs, T0s, R0s, Sensibilities)):
        R_bol, V_bol = get_Rbol_Vbol(channel, bias_V, V_I_path)
        if len(R_bol)==0:
            print('No Working point find')
            return None
        R_bols[i]= R_bol*1e-6
        eta =compute_reduced_sensibility(Sensibility, T0, R0, V_bol, R_bol)
        ethas[i] = eta
    ax.scatter(R_bols[channels645], ethas[channels645],label='UV645',s=100,marker='s')
    ax.scatter(R_bols[channels620], ethas[channels620], label='UV620',s=100,marker='v')
    ax.set_xlabel('Resistance (M$\Omega$)')
    ax.set_ylabel('Reduced Sensitivity $\eta$ ($GeV^{-1}$)')
    plt.legend(loc = 'upper left')

channels = [3,4,5,6,7,8,11,12]
Sensibilities = [5.08,58,75.7,54.3,5.8,73.4,108,86.2]
bias_Vs = 2 * np.array([3,0.5,0.3,0.5,4,0.5,0.5,0.5])
filename_VI = "15mK_2Gohm_all.xlsx"
V_I_path = "/Users/mp274748/Documents/data_arg/RUN97/I-V/" + filename_VI
plot_reduced_sensibility(channels, bias_Vs, T0s, R0s, Sensibilities,V_I_path)
plt.show()

