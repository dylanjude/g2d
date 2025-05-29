import numpy as np
import yaml, os, sys
from . import ogen, grid_utils, airfoil_utils
import matplotlib.pyplot as plt

def nicemesh(jtot,ktot,foil0,outfile,rounded=False):

    gen_i1 = { "ktot"       : int(ktot*0.75),   # points in normal dir
               "ds0"        : 6.6e-6, # wall spacing
               "far"        : 1.3,   # distance to far field
               "knormal"    : 10,     # points to walk straight out from wall
               "res_freq"   : 100,    # how often to check poisson solver residuals
               "omega"      : 1.1,    # SSOR relaxation constant (decrease if diverging)
               "initlinear" : 5.0     # High (~100) for simple geoms, low (~2) for concave geoms
           }
    gen_i2 = { "ktot"       : ktot-gen_i1['ktot'],   # points in normal dir
               "ds0"        : 0.05, # wall spacing
               "far"        : 30.0,   # distance to far field
               "knormal"    : 2,     # points to walk straight out from wall
               "res_freq"   : 100,    # how often to check poisson solver residuals
               "omega"      : 1.3,    # SSOR relaxation constant (decrease if diverging)
               "initlinear" : 80.0     # High (~100) for simple geoms, low (~2) for concave geoms
           }
    nlinear=20
    NP=600

    # try:
    if(1):
        # foil0 = load_airfoil(datfile)
        foil0 = airfoil_utils.close_te(foil0)
        foil  = airfoil_utils.interpolate(foil0,jtot,0.0015,nlinear,rounded)

        # --------------------------------
        # near-field region:
        ogen.set_ss_coeffs(0.2, 0.25, 0.2, 0.2) # normal@wall, space@wall, same for farfield
        gen   = ogen.MeshGen(foil, gen_i1)
        gen.poisson(NP)
        near  = gen.get_mesh()
        # grid_utils.plot_xy(near)

        # --------------------------------
        # far-field region:
        tmpf  = near[-1,:,:]
        ogen.set_ss_coeffs(0.3, 0.3, 0.3, 0.2) # normal@wall, space@wall, same for farfield
        gen2  = ogen.MeshGen(tmpf, gen_i2)
        gen2.poisson(100)
        far   = gen2.get_mesh()
        # grid_utils.plot_xy(far)

        # --------------------------------
        # combine the grids:
        full  = np.vstack([near,far[1:,:,:]])

        grid_utils.write_grid(outfile, full)
    # except:
    #     if(not rounded):
    #         return nicemesh(jtot,ktot,datfile,outfile,True)
    #     return 1
    return 0
