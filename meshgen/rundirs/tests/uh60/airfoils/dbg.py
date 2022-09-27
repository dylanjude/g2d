import numpy as np
import matplotlib.pyplot as plt

x1,y1 = np.loadtxt("sc1095.dat").T

x2,y2 = np.loadtxt("sc1094r8.dat").T


x3 = x2 * 1.009875
y3 = y2 * 1.009875

x3 = x3 - 0.007418


# x1 = x1 - 0.25
# x3 = x3 - 0.25

# x4 = x3
# y4 = y3

one = np.pi/180
x4  = x3*np.cos(one) - y3*np.sin(one)
y4  = x3*np.sin(one) + y3*np.cos(one)

delta = 0.5*(y4[0] + y4[-1])
y4 = y4 - delta
delta = 0.5*(x4[0] + x4[-1])
x4 = x4 - delta + 1.0


with open('sc1094r8_rot.dat', 'w') as f:
    for i in range(x4.shape[0]):
        f.write("%12.6f %12.6f\n"%(x4[i], y4[i]))


# plt.plot(x1,y1)
# plt.plot(x4,y4)


# plt.show()
