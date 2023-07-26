import numpy as np



def decode_binary(file_path, data_type=np.float16):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    return np.frombuffer(binary_data, dtype=data_type)


def plot_window(ax,plot_display,time, window_size,data,center=0.5):
    y_data = data[int(time-window_size*center):int(time+window_size*(1-center))]
    plot_display.set_ydata(y_data)
    plot_display.set_xdata(np.linspace(0,len(y_data),len(y_data)))
    print(np.min(y_data), np.max(y_data))
    ax.set_ylim(np.min(y_data), np.max(y_data))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    filename = dictionary['filename']
    data = decode_binary(path + filename[:-4] + ".bin")
    print(data)
    fig, ax = plt.subplots()
    plot_display,= ax.plot(0)
    print(len(data))
    print(len(data[data<1e10]))
    plot_window(ax,plot_display, len(data)//2, len(data)//2-1, data[np.data<1e10])
    plt.show()
