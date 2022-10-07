import os, numpy as np
from . import naca

plt = None

def make_naca(naca_string, jtot, blunt=True):
    if(jtot%2 == 0):
        jtot += 1
        print(("Jtot must be odd. I\'m adding 1. Jtot now = %d"%(jtot)))
    half         = int((jtot-1)/2)
    if(len(naca_string) == 5):
        x,y          = naca.naca5(naca_string, half, blunt, True)
    else:
        x,y          = naca.naca4(naca_string, half, blunt, True)
    x, y         = x[::-1], y[::-1]
    airfoil      = np.zeros((jtot,2))
    airfoil[:,0] = x
    airfoil[:,1] = y
    return airfoil

def load_grid(filename):
    threeD = False
    with open(filename, "r") as f:
        ln = f.readline().split()
        if(len(ln) == 2):
            jtot, ktot = [int(x) for x in ln]
        elif(len(ln) == 3):
            threeD = True
            jtot, ktot, ltot = [int(x) for x in ln]

    if(not threeD):
        data = np.loadtxt(filename, skiprows=1)
        data = data.reshape((2,ktot,jtot))
        reordered = np.zeros((ktot, jtot, 2))
        reordered[:, :, 0] = data[0, :, :]
        reordered[:, :, 1] = data[1, :, :]
    else:
        data = np.loadtxt(filename, skiprows=1)
        data = data.reshape((3,ltot,ktot,jtot))
        reordered = np.zeros((ltot, ktot, jtot, 3))
        reordered[:, :, :, 0] = data[0, :, :, :]
        reordered[:, :, :, 1] = data[1, :, :, :]
        reordered[:, :, :, 2] = data[2, :, :, :]
    return reordered

def write_grid(filename, xyz):
    with open(filename, 'w') as f:
        print((xyz.shape))
        if(len(xyz.shape) == 4):
            print("3D Grid")
            threeD = True
            ltot,ktot,jtot,nvar = xyz.shape
        elif(len(xyz.shape) == 3):
            print("2D Grid")
            threeD = False
            ltot = 1
            ktot,jtot,nvar = xyz.shape
        if(threeD):
            f.write("%d %d %d\n"%(jtot,ktot,ltot))
        else:
            f.write("%d %d\n"%(jtot,ktot))
        xyz = xyz.reshape((ltot,ktot,jtot,nvar))
        for var in range(nvar):
            for l in range(ltot):
                for k in range(ktot):
                    for j in range(jtot):
                        f.write("%25.16e\n"%(xyz[l,k,j,var]))
        print(("wrote %s"%filename))

def read_grid(filename):
    return load_grid(filename)

def plot_xy(lxy, patts=['-k', '-r', '-g', '-b', '-m']):
    global plt
    if(plt is None):
        from matplotlib import pyplot
        plt = pyplot
    patts = patts + patts + patts
    patts = patts[::-1]
    plt.figure(figsize=(9,9))
    lw = 1.2
    if(type(lxy) != list):
        lxy = [lxy]
    for xy in lxy:
        ktot, jtot, nv = xy.shape
        patt = patts.pop()
        if(nv != 2):
            print("incorrect number of vars")
            return
        for k in range(ktot):
            plt.plot(xy[k,:,0], xy[k,:,1], patt, lw=lw)
        for j in range(jtot):
            plt.plot(xy[:,j,0], xy[:,j,1], patt, lw=lw)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    
def plot_file(filename):
    with open(filename, "r") as f:
        jtot, ktot = [int(x) for x in f.readline().split()]
    data = np.loadtxt(filename, skiprows=1)
    data = data.reshape((2, ktot, jtot))
    reordered = np.zeros((ktot, jtot, 2))
    reordered[:, :, 0] = data[0, :, :]
    reordered[:, :, 1] = data[1, :, :]
    plot_xy(reordered)


from ..build.lib import libgen2d
from . import airfoil_utils
def generate_mesh(uiuc_coords_file, gridfile, doplot=False):
  jtot,ktot  = 301,122
  gen_inputs = { "ktot"       : ktot,   # points in normal dir               
                 "ds0"        : 6.6e-6, # wall spacing              
                 "far"        : 30.0,   # distance to far field              
                 "knormal"    : 10,     # points to walk straight out from wall       
                 "res_freq"   : 100,    # how often to check poisson solver residuals          
                 "omega"      : 1.4,    # SSOR relaxation constant (decrease if diverging)     
                 "initlinear" : 7.0     # High (~100) for simple geoms, low (~2) for concave geoms      
                } 
  nlinear  = 20
  rounded  = True
  np       = 600

  if doplot:
    gen_inputs['res_freq'] = 10

  print("Loading "+uiuc_coords_file)

  foil0 = airfoil_utils.load_uiuc(uiuc_coords_file)
  foil0 = airfoil_utils.close_te(foil0)
  foil  = airfoil_utils.interpolate(foil0,jtot,0.0015,nlinear,rounded)
  gen   = libgen2d.MeshGen(foil, gen_inputs)
  gen.poisson(np)
  if(doplot):
      xy = gen.get_mesh()
      plot_xy(xy)
      gen.write_to_file(gridfile)
      # continue
  else:
      gen.write_to_file(gridfile)
  print('current working directory is ',os.getcwd())
  print('wrote a grid file called ',gridfile)
  return None
