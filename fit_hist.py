import numpy as np
import matplotlib.pyplot as plt

#amp=np.array([170,211,231,346,384,435,643]) #for'/Users/mp274748/Documents/20180709_23h07.BINLMO.ntp'
amp=np.array([142,206,247,316,407,485,712])
value=np.array([238.6,295,352,510,583,609,911])

z=np.polyfit(amp,value,1)
p = np.poly1d(z)
plt.scatter(amp,value)
x = np.linspace(0, 1000, 700)
y = p(x)
print(z)
plt.plot(x, y, c='r')
plt.show()