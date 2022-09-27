import numpy as np
import scipy.interpolate
import scipy.optimize

DIST_EXPONENT = 80 # small ~18 means more points near tail. large means less

def get_distribution(dmax, dmin, length):
    walk   = 0.0
    t      = [0.0]
    i = 0
    while walk < length:
        ratio = walk * 1.0 / length
        delta = dmin + (1.0-np.cos(ratio * 2.0*np.pi)**80)*(dmax - dmin)
        walk = walk + delta
        t.append(walk)
        i+=1
    t[i] = length
    return np.array(t)
    
def match_jtot(jtot, length, dmin):

    dmax1 = 0.1
    dmax2 = dmin
    found = False

    tot  = get_distribution(dmax1, dmin, length).shape[0]
    if(tot > jtot):
        print(("high", tot))
        raise
    tot  = get_distribution(dmax2, dmin, length).shape[0]
    if(tot < jtot):
        print(("low", tot))
        raise

    i = 0
    while(not found):
        dmax = (dmax1 + dmax2)/2
        tot  = get_distribution(dmax, dmin, length).shape[0]
        if(tot == jtot):
            found = True
        elif(tot < jtot):
            dmax1 = dmax
        elif(tot > jtot):
            dmax2 = dmax
        i+=1
        if(i > 100):
            print("exiting")
            raise
    return dmax

def round_tail(airfoil, nadd=4):
    # put nose at 0 and scale
    # airfoil[:,0] = airfoil[:,0]-np.min(airfoil[:,0])
    # scale = 1.0 / np.max(airfoil[:,0])
    # airfoil[:,0] *= scale
    # airfoil[:,1] *= scale

    jtot = airfoil.shape[0]
    x    = airfoil[:,0]
    y    = airfoil[:,1]

    # make sure the trailing edge is closed:
    x[ 0]  = 0.5*(x[0] + x[-1])
    y[ 0]  = 0.5*(y[0] + y[-1])
    x[-1]  = x[0]
    y[-1]  = y[0]
    theta0 = np.arctan(0.5*(y[4]-y[0]+y[jtot-1-4]-y[0])/(x[4]-x[0]))
    i = 1
    # now open it up a little bit
    eps = 0.0015
    while(True):
    # while(abs(y[-1-i]-y[i]) < eps):
        dx = 0.5*(x[-1-i]-x[i])
        dy = 0.5*(y[-1-i]-y[i])
        dist = np.sqrt(dx*dx + dy*dy)
        if( dist > eps ):
            break
        # print "going to ", i
        dist    = eps/dist
        avgx    = 0.5*(x[-1-i]+x[i])
        avgy    = 0.5*(y[-1-i]+y[i])
        x[i]    = avgx - dist * dx
        x[-1-i] = avgx + dist * dx
        y[i]    = avgy - dist * dy
        y[-1-i] = avgy + dist * dy
        i+=1
        
    eps   = min(y[0]-y[1], y[-2]-y[-1])
    angle = 90.0

    jtot = x.size+nadd*2
    newx             = np.zeros(jtot)
    newy             = np.zeros(jtot)

    i = 1
    while(abs(x[0] - x[i]) < 2*eps):
        # print "adding 1 to add"
        nadd += 1
        i+= 1
    i -= 1

    # i = 0
    newx[nadd:-nadd-1] = x[i:-i-1]
    newy[nadd:-nadd-1] = y[i:-i-1]
    newx[0]            = x[0]
    newx[-1]           = x[-1]
    newy[0]            = y[0]
    newy[-1]           = y[-1]

    for j in range(1,nadd+1):
        dtheta_top = (angle*np.pi/180)/nadd
        dtheta_bot = (angle*np.pi/180)/nadd
        newx[j]        = x[0] - eps*(1.0 - np.cos(dtheta_bot*j))
        newx[jtot-1-j] = x[0] - eps*(1.0 - np.cos(dtheta_top*j))
        newy[j]        = y[0] - eps*(np.sin(dtheta_bot*j))
        newy[jtot-1-j] = y[0] + eps*(np.sin(dtheta_top*j))

    for j in range(1, nadd+1):
        tmpx, tmpy = newx[j]-newx[0], newy[j]-newy[0]
        newx[j] = newx[0] + tmpx*np.cos(theta0) - tmpy*np.sin(theta0)
        newy[j] = newy[0] + tmpx*np.sin(theta0) + tmpy*np.cos(theta0)
        tmpx, tmpy = newx[jtot-1-j]-newx[0], newy[jtot-1-j]-newy[0]
        newx[jtot-1-j] = newx[0] + tmpx*np.cos(theta0) - tmpy*np.sin(theta0)
        newy[jtot-1-j] = newy[0] + tmpx*np.sin(theta0) + tmpy*np.cos(theta0)

    airfoil      = np.zeros((newx.size,2))
    airfoil[:,0] = newx
    airfoil[:,1] = newy

    return airfoil

def interpolate(airfoil, jtot, dmin=0.003, nlinear=20, rounded=True):

    if(rounded):
        nadd = 4 # if dmin > 0.0029 else 6
        airfoil = round_tail(np.copy(airfoil))
    else:
        airfoil1 = np.copy(airfoil)
    x       = airfoil[:,0]
    y       = airfoil[:,1]

    # parameterize
    xd     = np.diff(x)
    yd     = np.diff(y)
    dist   = np.sqrt(xd**2+yd**2)
    u      = np.cumsum(dist)
    u      = np.hstack([[0],u])
    length = u.max()

    dmax = match_jtot(jtot,length,dmin)
    t    = get_distribution(dmax, dmin, length)
    
    fix = length - t[-2] - dmin
    t[1:-1] = t[1:-1]+fix/2

    xn = np.zeros_like(t)
    yn = np.zeros_like(t)

    # get rid of duplicates (some uiuc db meshes have duplicate pts!)
    ii, nu = 1, u.size
    while(ii<nu):
        if(u[ii]==u[ii-1]):
            u=np.delete(u,ii)
            x=np.delete(x,ii)
            y=np.delete(y,ii)
            nu=nu-1
        ii+=1

    if(nlinear > x.shape[0]):
        print("all linear")
        fx    = scipy.interpolate.interp1d(u,x,kind="linear")
        fy    = scipy.interpolate.interp1d(u,y,kind="linear")
        xn[:] = fx(t[:])
        yn[:] = fy(t[:])
        # xn[-nlinear:]  = fx(t[-nlinear:])
        # yn[:nlinear]   = fy(t[:nlinear] )
        # yn[:] = scipy.interpolate.pchip_interpolate(u,y,t[:])
        # xn[:] = scipy.interpolate.pchip_interpolate(u,x,t[:])
    else:
        # xn[:nlinear]   = scipy.interpolate.pchip_interpolate(u,x,t[:nlinear])
        # yn[:nlinear]   = scipy.interpolate.pchip_interpolate(u,y,t[:nlinear])
        # xn[-nlinear:]  = scipy.interpolate.pchip_interpolate(u,x,t[-nlinear:])
        # yn[-nlinear:]  = scipy.interpolate.pchip_interpolate(u,y,t[-nlinear:])
        # xn[:nlinear]   = scipy.interpolate.interp1d(u,x,kind="quadratic")(t[:nlinear] )
        # yn[:nlinear]   = scipy.interpolate.interp1d(u,y,kind="quadratic")(t[:nlinear] )
        # xn[-nlinear:]  = scipy.interpolate.interp1d(u,x,kind="quadratic")(t[-nlinear:])
        # yn[-nlinear:]  = scipy.interpolate.interp1d(u,y,kind="quadratic")(t[-nlinear:])
        fx = scipy.interpolate.interp1d(u,x,kind="linear")
        fy = scipy.interpolate.interp1d(u,y,kind="linear")
        xn[:nlinear]   = fx(t[:nlinear] )
        xn[-nlinear:]  = fx(t[-nlinear:])
        yn[:nlinear]   = fy(t[:nlinear] )
        yn[-nlinear:]  = fy(t[-nlinear:])
        # xn[nlinear:-nlinear] = scipy.interpolate.interp1d(u,x,kind="cubic")(t[nlinear:-nlinear])
        # yn[nlinear:-nlinear] = scipy.interpolate.interp1d(u,y,kind="cubic")(t[nlinear:-nlinear])
        xn[nlinear:-nlinear] = scipy.interpolate.interp1d(u[1:-1],x[1:-1],kind="cubic")(t[nlinear:-nlinear])
        yn[nlinear:-nlinear] = scipy.interpolate.interp1d(u[1:-1],y[1:-1],kind="cubic")(t[nlinear:-nlinear])

    # xn[:] = scipy.interpolate.interp1d(u,x,kind="cubic")(t[:])
    # yn[:] = scipy.interpolate.interp1d(u,y,kind="cubic")(t[:])
    
    new_airfoil      = np.zeros((xn.size,2))
    new_airfoil[:,0] = xn
    new_airfoil[:,1] = yn
    return new_airfoil

def close_te(airfoil):
    a0 = airfoil.copy()
    if(abs(a0[-1,1]-a0[0,1]) > 1e-6):
        a1 = np.zeros((a0.shape[0]+2,2))
        a1[1:-1,:] = a0[:,:]
        a0 = a1
    a0[0,:]  = 0.5*(a0[1,:]+a0[-2])
    a0[-1,:] = a0[0,:]
    a0[:,0] -= np.min(a0[:,0])
    a0[:,:] /= a0[0,0]
    return a0

import numpy as np
import yaml, os

def load_uiuc(fname):
    a0 = np.loadtxt(fname, skiprows=1)
    if(a0[-1,0] == a0[-2,0]):
        a0 = a0[:-1,:]
    a0 = a0[::-1,:]
    return a0

# def load_db(ndvar=7):
#     db = None
#     dbfile = "afdb.{:d}.yaml".format(ndvar)
#     if(os.path.exists(dbfile)):
#         with open(dbfile, 'r') as f:
#             db = yaml.safe_load(f)
#     return db