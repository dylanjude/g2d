import numpy as np
import sys
sys.path.append("../../build/lib/")
sys.path.append("../../python")
import libgen2d
import naca
import grid_utils, airfoil_utils
import close
from spandist import distribution
from matplotlib import pyplot as plt
print "ok"

jtot = 191    # wrap
ktot = 64     # normal
ltot = 100    # span

# use a 4 or 5 digit naca airfoil:
airfoil = grid_utils.make_naca('0012', jtot)

# # or interpolate from some other airfoil:
# coarse  = np.loadtxt("sc1095.dat")
# airfoil = airfoil_utils.interpolate(coarse, jtot)

inputs = { "ktot"     : ktot,
           "ds0"      : 1.0e-4,
           "stretch"  : 1.14,    # this determines how far away your mesh goes
           "res_freq" : 100,
           "omega"    : 1.4 }

gen = libgen2d.MeshGen(airfoil, inputs)
gen.poisson(500)
# gen.write_to_file("grid.xyz")
xy = gen.get_mesh()

# cut to these many k planes (normal-direction)
ktot = 48
xy = xy[:ktot,:,:]

# grid_utils.plot_xy(xy)
# quit()

# ktot      = 1
# ltot      = 130
# xy        = np.zeros((ktot,jtot,2))
# xy[0,:,:] = airfoil

# length of the blade
lblade      = 5.0
# my own distribution function ( spandist.py ):
#     distribution( root dz, tip dz, #sections )
#     returns distribution from 0 to 1
zdist       = distribution(0.008, 0.005, ltot-2)
extra       = np.zeros(ltot)
extra[1:-1] = zdist
extra[0 ]   = zdist[0]   # set the first spacing
extra[-1]   = zdist[-1]  # set the last spacing
extra[1 ]   = 0.002      # add an intermediate cut
extra[-2]   = 0.999      # add an intermediate cut
# apply the distribution (from 0-1) to the blade
zdist       = extra * lblade

xyz       = np.zeros((ltot,ktot,jtot,3))
for l in range(ltot):
    xyz[l,:,:,:2] = xy
    xyz[l,:,:,2]  = zdist[l]

print zdist.shape

# now close the ends (8 planes at root, 12 at tip)
full = close.close_both(8, 12, xyz)

# for a right-hand coordinate system, reverse the order of
# planes. also apply quarter chord offset and root cutout.
print "Rearranging..."
final = np.zeros_like(full)
final[:,:,:,0] = full[::-1,:,:,0] - 0.25
final[:,:,:,1] = full[::-1,:,:,2] + 1.0
final[:,:,:,2] = full[::-1,:,:,1]

# do the collective pitch
theta = 8.0*np.pi/180
ct    = np.cos(theta)
st    = np.sin(theta)

x,y,z = close.rotate(final[:,:,:,0],
                     final[:,:,:,1],
                     final[:,:,:,2],
                     0.0, 0.0, 0.0,         # rotate about this point
                     0.0, 1.0, 0.0, ct, st) # rotate about this axis

final[:,:,:,0] = x
final[:,:,:,1] = y
final[:,:,:,2] = z

print "Writing..."
grid_utils.write_grid("grid3d.xyz", final)


