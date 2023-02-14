import numpy as np
import sys
sys.path.append("/home/dylan/work/MeshGen/3doo/build/lib/")
sys.path.append("/home/dylan/work/MeshGen/3doo/python")
sys.path.append("/home/dylan/work/utils/python_cccgns")
import libgen2d
import libpycgns
import naca
import grid_utils, airfoil_utils
import close
from spandist import distribution
from matplotlib import pyplot as plt
import scipy.interpolate 
from inputs import *
print "ok"

print jtot,ktot,ltot

use_cgns = (len(sys.argv) == 1)

def linterp(rR, loc, dis):
    # return scipy.interpolate.interp1d(loc, dis, kind="linear", fill_value="extrapolate")(rR)
    if(rR < loc[0]):
        return dis[0]
    if(rR >= loc[-1]):
        return dis[-1]
    for i in range(len(loc)-1):
        if(rR > loc[i] and rR < loc[i+1]):
           frac = (rR-loc[i])/(loc[i+1]-loc[i])
           return dis[i] + frac*(dis[i+1]-dis[i])

def get_airfoil(rR, airfoil_r, airfoil):
    if(rR < airfoil_r[0]):
        return airfoil[0], airfoil[0], 1.0
    if(rR >= airfoil_r[-1]):
        return airfoil[-1], airfoil[-1], 1.0
    for i in range(len(airfoil_r)-1):
        if(rR > airfoil_r[i] and rR < airfoil_r[i+1]):
           frac = (rR-airfoil_r[i])/(airfoil_r[i+1]-airfoil_r[i])
           return airfoil[i], airfoil[i+1], frac

def generate_airfoil(name):
    airfoil0 = np.loadtxt(name)
    if(name.find("tab") >= 0):
        airfoil = airfoil_utils.interpolate(airfoil0, jtot, 0.002, 40)
    else:
        airfoil = airfoil_utils.interpolate(airfoil0, jtot, 0.001, 40)
    gen = libgen2d.MeshGen(airfoil, gen_inputs)
    gen.poisson(300)
    tmpxy = gen.get_mesh()
    del gen
    return tmpxy

ltot1 = ltot - close1 - close2
# ltot1 = 80

d1 = 0.003
d2 = 0.0009
rdist  = distribution(d1, d2, ltot1-2, 1.7)
if(rdist is None):
    quit("Could not do that spanwise distribution")
extra       = np.zeros(ltot1)
extra[1:-1] = rdist
extra[0 ]   = rdist[0]
extra[-1]   = rdist[-1]
extra[1 ]   = d1/4.0
extra[-2]   = 1.0-d2/4.0
rdist       = extra*(1-cutout)+cutout

# # simple
# rdist = np.linspace(0, 1, ltot1)*(1-cutout)+cutout

old_af1, old_af2 = "", ""
xy1, xy2         = None, None

xyz   = np.zeros((ltot1,ktot,jtot,3))
for l in range(ltot1):
    rr = rdist[l]
    #
    # Figure out the airfoil we're using
    #
    af1, af2, ratio = get_airfoil(rr, airfoil_r, airfoil)
    #
    # logic for airfoil 1
    if(af1 == old_af1):
        pass
    elif(af1 == old_af2):
        xy1 = xy2
    else:
        xy1 = generate_airfoil(af1)
    #
    # locig for airfoil 2
    if(af2 == old_af2):
        pass
    elif(af2 == af1):
        xy2 = xy1
    else:
        xy2 = generate_airfoil(af2)
    old_af1, old_af2 = af1, af2
    # interpolate airfoil
    xy = xy1 + ratio*(xy2 - xy1)
    # interpolate the other quantities:
    c  = linterp(rr, chord_r, chord)
    t  = linterp(rr, twist_r, twist)
    dc = linterp(rr, cloc_r, cloc)
    # twist
    ct = np.cos(-t)
    st = np.sin(-t)

    stretchx = linterp(rr, sx_r, sx)

    xy[:,:,:]  = c*(xy[:,:,:])     # scale the airfoil (still from 0->1)
    xy[:,:,0]  = stretchx*(xy[:,:,0])
    xy[:,:,0] += -0.25         # put the airfoil back along the reference quarter chord

    xyz[l,:,:,2] = rdist[l]*AR
    xyz[l,:,:,0] = xy[:,:,0]*ct - xy[:,:,1]*st
    xyz[l,:,:,1] = xy[:,:,0]*st + xy[:,:,1]*ct

    xyz[l,:,:,0] += dc                 # put the airfoil back along the reference quarter chord

# quit()

# print rdist.shape

xyz = close.close_both(close1, close2, xyz)
libgen2d.lsmooth(xyz[ :35,:,:,:], 0.7, True);
libgen2d.lsmooth(xyz[-70:,:,:,:], 0.7, True);

print "Rearranging..."
final = np.zeros_like(xyz)
final[:,:,:,0] =  xyz[::-1,:,:,2]
final[:,:,:,1] = -xyz[::-1,:,:,0]
final[:,:,:,2] =  xyz[::-1,:,:,1]

# match boeing root offset (correction for elastic axis location?)
final[:,:,:,1] += 0.08631344 
final[:,:,:,2] += -0.00365732

print "Writing cgns..."
libpycgns.write_cgns("blade.cgns", final)
if(not use_cgns):
    print "Writing xyz..."
    libgen2d.write_grid("grid3d.xyz", final)

# grid_utils.plot_xy(xy)

# mg.write_to_file('mesh.xyz')
# # grid_utils.plot_file('curved_ss.xyz')

