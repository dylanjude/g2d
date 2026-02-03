import numpy as np
import matplotlib.pyplot as plt

data  = np.loadtxt("higen_WRK/Slices_rotor_ccw_fine1_trm8.dat", comments=("A","--","#"))
twist = np.loadtxt("higen_WRK/Twist_rotor_ccw_fine1_trm8.dat", comments=("#"), skiprows=3)

last  = 0
foils = []
i     = 0

while i < data.shape[0]:
    i = last+1    
    while i < data.shape[0] and data[i,0]>data[i-1,0]:
        i+=1
    nspan = i-last
    foils.append(data[last:i,2:])
    last  = i

assert(len(foils) == twist.shape[0])

print("nfoils: ", len(foils))

# # plot all airfoils (no scaling)
# for i in range(twist.shape[0]):
#     f = foils[i]
#     tw = -twist[i,1]*np.pi/180
#     x = -f[:,0]*np.cos(tw) + f[:,1]*np.sin(tw)
#     y =  f[:,1]*np.cos(tw) + f[:,0]*np.sin(tw)
#     plt.plot(x,y,'-')

# plt.show()
# quit()


for i in range(twist.shape[0]):
# for i in [2,5,9,12]:
# for i in [5]:
    f = foils[i]

    # un-rotate the airfoil
    tw = -twist[i,1]*np.pi/180
    x  = -f[:,0]*np.cos(tw) + f[:,1]*np.sin(tw)
    y  =  f[:,1]*np.cos(tw) + f[:,0]*np.sin(tw)

    # un-scale the airfoil (0 < x < 1)
    x = x-np.min(x)
    scale = np.max(x)
    x = x/scale
    y = y/scale

    # put the leading edge at y=0
    ixmin = np.argmin(x)
    y = y-y[ixmin]

    # find location of trailing edge
    imax = np.argmax(x)

    # use temporary arrays where first/last point are not duplicated
    xtmp = x[:-1]
    ytmp = y[:-1]

    # "roll" the array so that the first point is at the maximum x-value
    xtmp = np.roll(xtmp, -imax)
    ytmp = np.roll(ytmp, -imax)

    # check that the airfoil is ordered clock-wise
    if(ytmp[5] > ytmp[-5]):
        xtmp = xtmp[::-1]
        ytmp = ytmp[::-1]

    # find the point where y crosses 0 and re-roll the airfoil so
    # that's the 0-index.
    yc=0
    while(ytmp[yc] < 0):
        yc-=1
    while(ytmp[yc] > 0):
        yc+=1
    xtmp = np.roll(xtmp, -yc)
    ytmp = np.roll(ytmp, -yc)

    # put the airfoil coords back in the full array with first/last duplicated
    x[:-1] = xtmp
    y[:-1] = ytmp
    x[-1]  = x[0]
    y[-1]  = y[0]

    # plt.plot(x,y,'-o',mfc="None",ms=8)
    # plt.plot(x[:8],y[:8],'-o',ms=8)
    # plt.plot(x[-3:],y[-3:],'-v')
    # plt.grid(True)
    # plt.show()

    fout = "airfoils/foil_{:02d}.dat".format(i)
    print("writing: {:s}".format(fout))
    with open(fout, "w") as f:
        for j in range(x.shape[0]):
            f.write("{:14.8f} {:14.8f}\n".format(x[j],y[j]))
# for i in range(x.shape[0]):
#     print("{:14.8f} {:14.8f}".format(x[i],y[i]))
    


        

# print(data.shape)
# print(data[:252])
