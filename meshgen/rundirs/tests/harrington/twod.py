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

use_cgns = (len(sys.argv) == 1)

name = airfoil[1]

blunt = True
foil = grid_utils.make_naca(name, jtot, blunt)

# initialize the mesh (do not run any Poisson smoothing). After this
# step you can get and plot the mesh to make sure there are no
# overlapping cells. If so, alter params in the inputs.
gen = libgen2d.MeshGen(foil, gen_inputs)

# # Once the initial mesh is good, run poisson smoothing to make the
# # mesh good for CFD.
gen.poisson(300)

xy = gen.get_mesh()

# mg.write_to_file('mesh.xyz')

grid_utils.plot_xy(xy)


