import numpy as np
import sys, os
import matplotlib.pyplot as plt
import yaml
sys.path.append('/home/djude/codes/g2d/build/lib')
from meshgen import utils

dir0   = "airfoils"
dir1   = "meshes"

if(len(sys.argv)>1):
    airfoils = [os.path.basename(sys.argv[1])]
else:
    airfoils = os.listdir(dir0)

rounded_te = False

# Modify this function so that your airfoil format is read correctly.
# foil surface must be of shape [nj,2] where nj is the number of
# wrap-around points. The airfoil must also start from the trailing
# edge and go from the bottom to the top in a clock-wise
# direction. The leading edge of the airfoil should be at x=0 and the
# trailing edge should be at 1.
def load_airfoil(fname):         
    a0 = np.loadtxt(fname)
    print(a0.shape)
    minx = np.min(a0[:,0])               
    maxx = np.max(a0[:,0])               
    a0[:,0]  = (a0[:,0]+minx)  
    a0[:,:] /= (maxx-minx)               
    if(a0[3,1]>a0[-3,1]):                
        a0 = np.ascontiguousarray(a0[::-1,:])            
    return a0   

i=0
for filename in airfoils:

    out = os.path.join(dir1,filename[:-3]+'x')

    if(os.path.exists(out)):
        print("  ->  Skipping {:s}, {:s} exists".format(filename,out))
        continue

    print("Loading "+filename)

    surf  = load_airfoil(os.path.join(dir0,filename))

    error = utils.nicemesh(301,140,surf,out,rounded_te)

    if error:
        print(filename + " failed")
