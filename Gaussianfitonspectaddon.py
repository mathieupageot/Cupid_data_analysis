import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
paraRAW = np.array([[5.66071043e+02, 3.07109997e+00],
       [9.83517615e+02, 2.36422438e+00],
       [1.48779372e+03, 2.20968173e+00],
       [2.60922372e+03, 3.64256070e+00]])
errorRAW = np.array([[0.26900705, 0.34396772],
       [0.18938959, 0.21442578],
       [0.29454137, 0.33068956],
       [0.14749021, 0.16128023]])
paraDIF = np.array([[ 566.52240791,    3.27749317],
       [ 980.96676946,    3.31896503],
       [1483.44084076,    3.66735382],
       [2609.24261954,    5.35718058]])
errorDIF = np.array([[0.18014076, 0.23480654],
       [0.18868016, 0.22137416],
       [0.4211806 , 0.49145635],
       [0.18585467, 0.21602787]])
ax.errorbar(paraRAW[:,0],paraRAW[:,1],errorRAW[:,0],errorRAW[:,1],label='RAW', linestyle='', barsabove=True, capsize=3, marker='x')
ax.errorbar(paraDIF[:,0],paraDIF[:,1],errorDIF[:,0],errorDIF[:,1],label='DIF', linestyle='', barsabove=True, capsize=3, marker='x')
ax.set_xlabel('Energy in keV')
ax.set_ylabel('Sigma in keV')
legend = fig.legend(loc='upper right', fontsize=20)

plt.show()