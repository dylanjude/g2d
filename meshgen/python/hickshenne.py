import numpy as np
from matplotlib import pyplot as plt
from naca import *

def hickshenne(n, xmi, w, SF, mag):
    x=np.linspace(0.0, 1.0, n, dtype='d')
    print(x)
    m = np.log(0.5)/np.log(xmi)
    b = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(m)):
            b[i] = b[i]+SF*mag[j]*np.sin(np.pi*x[i]**m[j])**w
    return b

def hh1(x, y, xmi, w, mag):
    m = np.log(0.5)/np.log(xmi)
    yy = np.copy(y)
    for i in range(len(x)):
        for j in range(len(m)):
            yy[i] = yy[i]+mag[j]*np.sin(np.pi*x[i]**m[j])**w
    return yy

def perturb(airfoil, design_vars):
    x = airfoil[:,0]
    y = airfoil[:,1]
    half = (x.shape[0]-1)/2

    xl = x[:half]
    xu = x[half:]
    yl = y[:half]
    yu = y[half:]

    locl = design_vars[0,:]
    locu = design_vars[1,:]
    magl = design_vars[2,:]
    magu = design_vars[3,:]
    w    = 2.0

    yy   = np.zeros_like(y)

    yl1 = hh1(xl,yl,locl,w,magl)
    yu1 = hh1(xu,yu,locu,w,magu)
    
    yy[:half] = yl1
    yy[half:] = yu1

    new_airfoil = np.copy(airfoil)
    new_airfoil[:,1] = yy
    return new_airfoil


if __name__ == "__main__":

    half = 63
    x, y = naca4('0012', half, True, True)
    # y[0], y[-1] = 0.0, 0.0
    n    = half*2 + 1

    print(len(x), n)
    print(half)
    
    xu = x[:half]
    xl = x[half:]
    yu = y[:half]
    yl = y[half:]
    
    locu = np.array([0.25, 0.5, 0.75])
    locl = np.array([0.25, 0.5, 0.75])
    magu = np.array([0.0, 0.0, 0.0])
    magl = np.array([0.0, 0.0, 0.0])
    w   = 2.0
    
    yy = np.zeros_like(y)
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(x,y,'--k', lw=2.0, label="original")
    
    for i in range(3):
        locu +=  0.1*(2*np.random.rand(3)-1)
        locl +=  0.1*(2*np.random.rand(3)-1)
        magu += 0.01*(2*np.random.rand(3)-1)
        magl += 0.01*(2*np.random.rand(3)-1)
    
        print('loc: ', locu, locl)
        print('mag: ', magu, magl)
        print('---')
        
        yu1 = hh1(xu,yu,locu,w,magu)
        yl1 = hh1(xl,yl,locl,w,magl)
    
        yy[:half] = yu1
        yy[half:] = yl1
    
        s = "random "+str(i)
        plt.plot(x,yy,'-o', lw=1.5, label=s)
    
    # plotting
    plt.axis([0, 1.0, -0.2, 0.2])
    plt.xlabel("Chord")
    plt.ylabel("Thickness")
    plt.grid('on')
    plt.legend(loc="lower right")
    #plt.savefig("hicks.pdf")
    plt.show()
    
    # mag = [0.01]
    
    
    # xmi = [0.5]
    # w   = 1.0
    # mag = [-0.01]
    # yl1 = hh1(xl,yl,xmi,w,mag)
    
    
    
    # plt.plot(x, y, '-b', lw=1.5)
    # plt.plot(xu, yu1, '-r', lw=1.5)
    # plt.plot(xl, yl1, '-g', lw=1.5)
    # plt.show()
    
    
    # b = hickshenne(100, np.array([0.1, 0.2, 0.3]), 2.0, 1.0, np.array([.1, .1, .1]))
    # print b
    
    # x = np.arange(len(b))
    
    # plt.plot(x,b)
    # plt.show()


