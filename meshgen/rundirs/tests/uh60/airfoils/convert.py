import numpy as np


xu, yu, xl, yl = np.loadtxt("sc1095_raw.txt").T

xl = xl[::-1]
yl = yl[::-1]

half = xu.shape[0]

x = np.zeros(half*2-1)
y = np.zeros(half*2-1)

x[0:half] = xl[:half]
x[half:]  = xu[1:]
y[0:half] = yl[:half]
y[half:]  = yu[1:]

with open('sc1095.dat', 'w') as f:
    for i in range(x.shape[0]):
        f.write("%12.6f %12.6f\n"%(x[i], y[i]))


xu, yu, xl, yl = np.loadtxt("sc1094r8_raw.txt").T

xl = xl[::-1]
yl = yl[::-1]

half = xu.shape[0]

x = np.zeros(half*2-1)
y = np.zeros(half*2-1)

x[0:half] = xl[:half]
x[half:]  = xu[1:]
y[0:half] = yl[:half]
y[half:]  = yu[1:]

with open('sc1094r8.dat', 'w') as f:
    for i in range(x.shape[0]):
        f.write("%12.6f %12.6f\n"%(x[i], y[i]))
