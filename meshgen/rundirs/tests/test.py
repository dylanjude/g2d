import numpy as np
import sys
sys.path.append('../build/lib')
sys.path.append('../python')
import libgen2d
import naca


jtot = 121
ktot = 60

if(jtot%2==0):
    quit('jtot must be odd')

half         = (jtot-1)/2
# x, y         = naca.naca4('0012', half, True, True)
x, y         = naca.naca5('25112', half, True, True)
x, y         = x[::-1], y[::-1]
airfoil      = np.zeros((jtot,2))
airfoil[:,0] = x
airfoil[:,1] = y

inputs = { "ktot"     : ktot,
           "ds0"      : 0.001,
           "stretch"  : 1.12,
           "res_freq" : 20,
           "omega"    : 1.5 }

gen = libgen2d.MeshGen(airfoil, inputs)
gen.poisson(100)
gen.write_to_file("grid.xyz")
