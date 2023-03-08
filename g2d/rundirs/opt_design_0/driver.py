import sys
sys.path.append('/home/dylan/work/codes/g2d/build/lib')
import garfoil
import numpy as np

machs = np.array([0.2, 0.3],'d')
aoas  = np.array([1.0, 3.0],'d')
reys  = np.array([1000000] ,'d')

foilname = "naca0012"
with open(foilname+".xyz","r") as f:
    jtot, ktot = [int(x) for x in f.readline().split()]
xy = np.loadtxt(foilname+".xyz",skiprows=1).reshape(2,ktot,jtot)
xy = xy.transpose((1,2,0))
xy = np.ascontiguousarray(xy)

garfoil.run(machs,reys,aoas,xy,"euler",foilname,3)
