import numpy as np
from matplotlib import pyplot as plt


def distribution(dmin1, dmin2, nz, exponent=4):

    dmax_high = 0.3
    dmax_low  = min(dmin1, dmin2)

    found = False
    i   = 0

    while(not found):
        dmax = 0.5*(dmax_high + dmax_low)
        # print "trying dmax = ", dmax
        loc   = 0.0
        zdist = [0.0]
        while loc < 1.0:
            ratio = loc
            step  = dmin1 + ratio*(dmin2-dmin1) + dmax*np.sin(np.pi * ratio)**exponent
            loc  += step
            zdist.append(loc)
            nl = len(zdist)
        if(nl == nz):
            found = True
        elif(nl > nz): # dmax is too low
            dmax_low = dmax
        elif(nl < nz): # dmax is too high
            dmax_high = dmax
        if(i > 50):
            return None
        if(i % 5 == 0):
            print(dmax_low, dmax_high)
        i+=1
    print("dmax is ", dmax)
    zdist = np.array(zdist)
    zdist = zdist / zdist[-1]
    return zdist
