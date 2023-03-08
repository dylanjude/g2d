import numpy as np
import sys
sys.path.append("/home/dylan/work/MeshGen/2D_python_ggen/build/lib/")
sys.path.append("/home/dylan/work/MeshGen/2D_python_ggen/python")
import libgen2d
import naca
import grid_utils
import close
from matplotlib import pyplot as plt
print("ok")

def write_xyz(filename, xyz):
    with open(filename, 'w') as f:
        print(xyz.shape)
        ltot,ktot,jtot,nvar = xyz.shape
        f.write("%d %d %d\n"%(jtot,ktot,ltot))
        for var in range(nvar):
            for l in range(ltot):
                for k in range(ktot):
                    for j in range(jtot):
                        f.write("%25.16e\n"%(xyz[l,k,j,var]))
        print("wrote %s"%filename)


jtot = 181
ktot = 84

airfoil = grid_utils.make_naca('0012', jtot)

inputs = { "ktot"     : ktot,
           "ds0"      : 0.00001,
           "stretch"  : 1.17,
           "res_freq" : 100,
           "omega"    : 1.5 }

gen = libgen2d.MeshGen(airfoil, inputs)
gen.poisson(200)
# gen.write_to_file("grid.xyz")
xy = gen.get_mesh()

ktot = 60
xy = xy[:ktot,:,:]

# grid_utils.plot_xy(xy)
# quit()

nz     = 80
lblade = 5.0
zdist = (np.cos(np.linspace(np.pi,2.0*np.pi,nz))+1.0)/2.0 * lblade

print(zdist.shape)

ltot = zdist.shape[0]

xyz = np.zeros((ltot,ktot,jtot,3))

for l in range(ltot):
    xyz[l,:,:,:2] = xy
    xyz[l,:,:,2]  = zdist[l]

# full = xyz
full = close.close_both(10, 10, xyz)

print("Rearranging...")
final = np.zeros_like(full)
final[:,:,:,0] = full[::-1,:,:,0] - 0.25
final[:,:,:,1] = full[::-1,:,:,2] + 1.0
final[:,:,:,2] = full[::-1,:,:,1]

theta = 8.0*np.pi/180
ct    = np.cos(theta)
st    = np.sin(theta)

x,y,z = close.rotate(final[:,:,:,0],
                     final[:,:,:,1],
                     final[:,:,:,2],
                     0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, ct, st)

final[:,:,:,0] = x
final[:,:,:,1] = y
final[:,:,:,2] = z

print("Writing...")
write_xyz("grid3d.xyz", final)

# grid_utils.plot_xy(xy)

# mg.write_to_file('mesh.xyz')
# # grid_utils.plot_file('curved_ss.xyz')

