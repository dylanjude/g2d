import numpy as np

jtot   = 191
ktot   = 84
ltot   = 220
close1 = 8
close2 = 12
AR     = 15.5
cutout = 0.131
lblade = (1-cutout)*AR

gen_inputs = { "ktot"     : ktot,
               "ds0"      : 5.0e-6,
               "far"      : 1,
               "knormal"  : 20,
               "res_freq" : 100,
               "omega"    : 1.4
           }

airfoil_r = [ 0.1925,
              0.4658,
              0.4969,
              0.7216,
              0.7316,
              0.8230,
              0.8540,
              0.8629,
              0.8729,
              1.0   ]

# 0.1925   sc1095        
# 0.4658   sc1095        
# 0.4969   sc1094r8      
# 0.7216*  sc1094r8      
# 0.7316   sc1094r8_tab  
# 0.8230   sc1094r8_tab  
# 0.8540   sc1095_tab    
# 0.8629   sc1095_tab    
# 0.8729*  sc1095        


airfoil   = ['airfoils/sc1095_closed.dat',
             'airfoils/sc1095_closed.dat',
             'airfoils/sc1094r8_rot_closed.dat',
             'airfoils/sc1094r8_rot_closed.dat',
             'airfoils/sc1094r8_rot_tab.dat',
             'airfoils/sc1094r8_rot_tab.dat',
             'airfoils/sc1095_tab.dat',
             'airfoils/sc1095_tab.dat',
             'airfoils/sc1095_closed.dat',
             'airfoils/sc1095_closed.dat']

chord_r = [0.2, 1.0]
chord   = [1.0, 1.0]

twist_r = [ 0.1322222, 
            0.1855556, 
            0.2166667, 
            0.8544444, 
            0.8688889, 
            0.8988889, 
            0.9288889, 
            0.9544444, 
            0.9666667, 
            0.9816667, 
            0.9955556 ]

twist = [  9.658110 ,
           9.657247 ,
           9.350968 ,
           -1.168977,
           -1.282461,
           -2.641949,
           -3.639037,
           -3.345000,
           -2.869546,
           -2.167637,
           -1.363785 ]

twist   = np.array(twist)*np.pi/180


# chord in swept region increases by 1.06418

cloc_r  = [ 0.2   ,  #
            0.9286,
            0.9507,
            1.0   ]


cloc    = [ 0.0,   
            0.0,
            0.0603,
            0.3389] 

sx_r = [ 0.2   ,  #
         0.9286,
         0.9507,
         1.0 ]

sx = [1.0, 
      1.0,
      1.063,
      1.063]


# ox_r = [0.0,  0.9286, 0.9507,  1.0000]
# ox   = [0.0,  0.0000, 0.0603,  0.3389]

# ?
oy_r = [ 0.0,    0.9489,      1.0]
oy   = [ 0.0,    0.00,    -0.2691]
