import numpy, pickle
import matplotlib.pyplot as plt
import xfoil_wrapper, xfoil_utilities
from scipy.optimize import least_squares

# residual function
def residual(x, *args):
    airfoil = args[0]
    params  = args[1]
    xy_orig = airfoil['xfoil_xy']
    Nupper  = params['Nupper']
    method  = params['method']

    if method.upper() == 'HHB':
        params['upper']['mag'] = x[0:Nupper]
        params['lower']['mag'] = x[Nupper:]
    elif method.upper() == 'CST':
        params['upper']['mag'] = x[0:Nupper]
        params['lower']['mag'] = x[Nupper:]
    elif method.lower() == 'bezier':
        params['upper']['P'] = x[0:Nupper]
        params['lower']['P'] = x[Nupper:]
    elif method.lower() == 'parsec':
        params['upper']['mag'] = x[0:Nupper]
        params['lower']['mag'] = x[Nupper:]
    elif method.lower() in ['polyfoil','naca']:
        params['upper']['alpha'] = x[0:Nupper]
        params['lower']['alpha'] = x[Nupper:]
    else:
        quit('unknown method')
    xy,_    = xfoil_utilities.airfoil_coords(airfoil, params, superimpose=False)
    res     = numpy.sum(numpy.square(xy[:,1] - xy_orig[:,1]))
    return res

#---------------------------------------------------------------
# load airfoils one by one
dir0 = "airfoils"
dir1 = "parameterized"

db = {}
dbfile = "afdb.{:d}.yaml".format(ndvar)
if(os.path.exists(dbfile)):
    with open(dbfile, 'r') as f:
        db = yaml.safe_load(f)
    print("loaded database",len(db.keys()))

i=0

for filename in os.listdir(dir0):
    # if(i>1):
    #     break
    # i+=1

    newfname = "{:s}.{:d}.cst".format(filename[:-4],ndvar)

    if(filename in db.keys()):
        print("  ->  Skipping {:s}, {:s} exists".format(filename,newfname))
        continue

    print("Loading "+filename)

    a0 = utils.load_uiuc(dir0+"/"+filename)

    dvars    = kulfan.match_airfoil(a0,nw,quiet=False)
    x1,y1    = kulfan.make_airfoil(dvars,a0[:,0])

    db[filename] = dvars.tolist()

    if(len(dvars) != ndvar):
        quit("problem")

    # print("Writing {:s}".format(out))
    # with open(out, 'w') as f:
    #     for ii in range(len(dvars)):
    #         f.write("{:16.8e} ".format(dvars[ii]))
    #     f.write("\n")

# print(d)

with open(dbfile, "w") as f:
    yaml.dump(db, f, default_flow_style=False)


# plt.plot(a0[:,0],a0[:,1],'-o')
# plt.plot(x1,y1,'-o')

# plt.grid(True)
# plt.show()

# get reference airfoil shape: NACA 0012
xy_orig   = xfoil_utilities.load_airfoil('dna02.dat')
adict   = {'Ref': xy_orig['xfoil_xy']}

#==========================================================================================
# CST: generate initial guess
#==========================================================================================

# CST   = xfoil_utilities.CST_guess()
# x0      = numpy.hstack((CST['upper']['mag'], CST['lower']['mag']))
# args    = (xy_orig, CST, )
# # setup least squares problem 
# lb      = numpy.ones_like(x0)*(-0.5)
# ub      = numpy.ones_like(x0)*( 2.0)

# sol     = least_squares(residual, x0, bounds=(lb, ub), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, \
#               loss='linear', f_scale=1.0, diff_step=None, max_nfev=20000, verbose=2, args=args)

# # get coordinates for "fitted" airfoil
# CST['upper']['mag'] = sol.x[:len(CST['upper']['mag'])]
# CST['lower']['mag'] = sol.x[len(CST['upper']['mag']):]
# xy,_      = xfoil_utilities.airfoil_coords(xy_orig, CST, superimpose=False)
# adict['CST'] = numpy.copy(xy)
# with open('fitted_'+CST['method']+'.pkl','wb') as f:
#     pickle.dump(sol,f,protocol=2)

#==========================================================================================
# plot airfoils
#==========================================================================================
# adict   = {'0012': xy_orig['xfoil_xy'], 'PARSEC': xy3}
xfoil_utilities.plot_airfoil(adict,overlay=True)

# save TXT file in plain file format
numpy.savetxt('fitted_foil.txt',xy)