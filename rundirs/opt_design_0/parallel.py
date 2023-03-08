import sys,os
sys.path.append('/p/home/djude/codes/g2d/build/lib')
import garfoil
import numpy as np
import mpi4py.MPI as MPI

machs = np.array([0.2, 0.4, 0.6],'d')
aoas  = np.arange(-10.0, 22.0, 2.0)
# machs = np.array([0.2],'d')
# aoas  = np.array([1.0],'d')
reys  = np.array([1e6] ,'d')

foils = ["naca0015","vr12","sc1095","rc410","vr13","clarky"]

rank = MPI.COMM_WORLD.Get_rank()

foilname = foils[rank]
foilfile = "uiuc_p3d_meshes/"+foilname+".x" 
with open(foilfile,"r") as f:
    jtot, ktot = [int(x) for x in f.readline().split()]
xy = np.loadtxt(foilfile,skiprows=1).reshape(2,ktot,jtot)
xy = xy.transpose((1,2,0))
xy = np.ascontiguousarray(xy)

garfoil.run(machs,reys,aoas,xy,"sa",foilname,3,rank)

MPI.COMM_WORLD.Barrier()
