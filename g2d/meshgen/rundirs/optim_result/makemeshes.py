import sys, os
from g2d.meshgen.python import grid_utils
airfoils = sys.argv[1:]  
for filename in airfoils:
    gridfile = filename.replace('xfoil_xy','grid')
    grid_utils.generate_mesh(filename, gridfile, doplot=True)
