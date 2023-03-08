import sys,os
from g2d.build.lib import garfoil
import numpy as np

#aoas  = np.linspace(-5,16,22,dtype='d')
aoas = np.linspace(-10.0,20.0,31,dtype='d') #[-2.0,-1.0]
#for iaoa in range(16):
#    aoa_list.append(float(iaoa))
#aoas  = np.array(aoa_list,'d')
reys  = np.array([2e6] ,'d')
machs = np.array([0.15],'d')

foilname = 'opt_design_1.grid' #sys.argv[1]
with open(foilname,"r") as f:
    jtot, ktot = [int(x) for x in f.readline().split()]
xy = np.loadtxt(foilname,skiprows=1).reshape(2,ktot,jtot)
xy = xy.transpose((1,2,0))
xy = np.ascontiguousarray(xy)

rank = 0
#garfoil.run(machs,reys,aoas,xy,"euler",foilname.replace('.grid',''),3)
# use the line below
garfoil.run(machs,reys,aoas,xy,"sa",foilname.replace('.grid',''),5,rank)
