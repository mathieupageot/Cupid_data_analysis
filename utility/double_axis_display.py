def make_format(current, other):
    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Down: {:<40}    Up: {:<}'
                .format(*['({:.4f}, {:.4f})'.format(x, y) for x,y in coords]))
    return format_coord
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(6)
    numdata = 100
    t = np.linspace(0.05, 0.11, numdata)
    y1 = np.cumsum(np.random.random(numdata) - 0.5) * 40000
    y2 = np.cumsum(np.random.random(numdata) - 0.5) * 0.002

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax2.format_coord = make_format(ax2, ax1)

    ax1.plot(y1, t, 'r-', label='y1')
    ax2.plot(y2, t, 'g-', label='y2')

    plt.show()
