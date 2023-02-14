import numpy as np

jtot   = 191
ktot   = 84
ltot   = 220
close1 = 12
close2 = 12
AR     = 8.3333
cutout = 0.2
lblade = (1-cutout)*AR

gen_inputs = { "ktot"     : ktot,
               "ds0"      : 5.0e-6,
               "far"      : 1,
               "knormal"  : 30,
               "res_freq" : 100,
               "omega"    : 1.1
           }

airfoil_r = [ 0.16666, 1.0]

airfoil   = ['0028','0015']

chord_r = [0.2, 1.0]
chord   = [1.0, 1.0]

twist_r = [ 0.1, 1.0]

twist = [  0.0, 0.0 ]

twist   = np.array(twist)*np.pi/180

# chord in swept region increases by 1.06418

cloc_r  = [ 0.1   , 1.0]

cloc    = [ 0.0,  0.0 ]

sx_r = [ 0.1, 1.0]

sx = [1.0, 1.0]

# ox_r = [0.0,  0.9286, 0.9507,  1.0000]
# ox   = [0.0,  0.0000, 0.0603,  0.3389]

# ?
# oy_r = [ 0.0,    0.9489,      1.0]
# oy   = [ 0.0,    0.00,    -0.2691]
