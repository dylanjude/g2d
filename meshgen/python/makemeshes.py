import numpy as np
import sys, os
import matplotlib.pyplot as plt
import yaml
import utils
sys.path.append('/home/judedp/codes/3doo/build/lib')
sys.path.append('/home/judedp/codes/3doo/python')
import libgen2d, grid_utils, airfoil_utils

dir0   = "airfoils"
dir1   = "meshes"

if(len(sys.argv)>1):
    airfoils = [os.path.basename(sys.argv[1])]
else:
    airfoils = os.listdir(dir0)

i=0
for filename in airfoils:

    out = os.path.join(dir1,filename[:-3]+'x')

    # if(os.path.exists(out)):
    #     print("  ->  Skipping {:s}, {:s} exists".format(filename,out))
    #     continue

    print("Loading "+filename)

    error = utils.nicemesh(301,140,filename)

    if error:
        print(filename + " failed")
