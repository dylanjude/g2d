import numpy as np
from matplotlib import pyplot as plt


def distribution(dmin1, dmin2, nz):

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
            step  = dmin1 + ratio*(dmin2-dmin1) + dmax*np.sin(np.pi * ratio)**4
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
        i+=1
    print "dmax is ", dmax
    zdist = np.array(zdist)
    zdist = zdist / zdist[-1]
    return zdist
    

# zdist = distribution(0.01, 0.03, 40)

# print zdist.shape
# # 

# # tmpy = np.diff(zdist)

# # tmpy = zdist
# # tmpx = np.arange(tmpy.shape[0])

# tmpy = np.zeros_like(zdist) + 0.4
# tmpx = zdist

# plt.plot(tmpx, tmpy, '-o', mfc="None")
# plt.show()
# quit()



# ktot = 1
# xyz  = np.zeros((ltot,ktot,jtot,3))


# print zdist.shape

# ltot = zdist.shape[0]

# xyz = np.zeros((ltot,ktot,jtot,3))

# for l in range(ltot):
#     xyz[l,:,:,:2] = xy
#     xyz[l,:,:,2]  = zdist[l]

# # full = xyz
# full = close.close_both(10, 10, xyz)

# print "Rearranging..."
# final = np.zeros_like(full)
# final[:,:,:,0] = full[::-1,:,:,0] - 0.25
# final[:,:,:,1] = full[::-1,:,:,2] + 1.0
# final[:,:,:,2] = full[::-1,:,:,1]

# theta = 8.0*np.pi/180
# ct    = np.cos(theta)
# st    = np.sin(theta)

# x,y,z = close.rotate(final[:,:,:,0],
#                      final[:,:,:,1],
#                      final[:,:,:,2],
#                      0.0, 0.0, 0.0,
#                      0.0, 1.0, 0.0, ct, st)

# final[:,:,:,0] = x
# final[:,:,:,1] = y
# final[:,:,:,2] = z

# print "Writing..."
# grid_utils.write_grid("grid3d.xyz", final)

# # grid_utils.plot_xy(xy)

# # mg.write_to_file('mesh.xyz')
# # # grid_utils.plot_file('curved_ss.xyz')

