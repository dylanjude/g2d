import sys
import numpy as np
from matplotlib import pyplot
sys.path.append("/home/dylan/work/MeshGen/3doo/build/lib/")
sys.path.append("/home/dylan/work/MeshGen/3doo/python")
sys.path.append("/home/dylan/work/utils/python_cccgns")
import libgen2d
import libpycgns

if(len(sys.argv) < 2):
    quit("enter the cgns blade filename")

if(len(sys.argv) < 3):
    quit("enter the angle to rotate")

cgns_file = sys.argv[1]
theta     = sys.argv[2]
suffix    = cgns_file.find('.cgns')

if(suffix < 0):
    quit("CGNS extension incorrect")

xyz_file = cgns_file[:suffix]+"{:s}.xyz".format(theta)

theta = float(theta)
print "Rotating by {:f} degrees".format(theta)

sol = libpycgns.CGNSWrap(cgns_file)
x   = sol.get_x()
y   = sol.get_y()
z   = sol.get_z()
ltot,ktot,jtot = x.shape

pi, cos, sin, sqrt = np.pi, np.cos, np.sin, np.sqrt

# rotate the point x,y,z about the vector u,v,w that passes through a,b,c
def rotate(x,y,z,a,b,c,u,v,w,ct,st):
    x1 = (a*(v*v+w*w)-u*(b*v+c*w-u*x-v*y-w*z))*(1.0-ct)+x*ct+(-c*v+b*w-w*y+v*z)*st
    y1 = (b*(u*u+w*w)-v*(a*u+c*w-u*x-v*y-w*z))*(1.0-ct)+y*ct+( c*u-a*w+w*x-u*z)*st
    z1 = (c*(u*u+v*v)-w*(a*u+b*v-u*x-v*y-w*z))*(1.0-ct)+z*ct+(-b*u+a*v-v*x+u*y)*st
    return x1,y1,z1

#
# Pitch
#
a, b, c = 0.0, 0.0, 0.0  # point
u, v, w = 1.0, 0.0, 0.0  # axis 

theta = theta*pi/180
ct    = cos(theta)
st    = sin(theta)

print "pitching..."

x1, y1, z1 = rotate(x,y,z,a,b,c,u,v,w,ct,st)

xyz = np.zeros((ltot,ktot,jtot,3))
xyz[:,:,:,0] = x1
xyz[:,:,:,1] = y1
xyz[:,:,:,2] = z1

# print "Writing cgns..."
# libpycgns.write_cgns("blade_rot.cgns", xyz)

print xyz_file
libgen2d.write_grid(xyz_file, xyz)
