import numpy as np

# rotate the point x,y,z about the vector u,v,w that passes through a,b,c
def rotate(x,y,z,a,b,c,u,v,w,ct,st):
    x1 = (a*(v*v+w*w)-u*(b*v+c*w-u*x-v*y-w*z))*(1.0-ct)+x*ct+(-c*v+b*w-w*y+v*z)*st
    y1 = (b*(u*u+w*w)-v*(a*u+c*w-u*x-v*y-w*z))*(1.0-ct)+y*ct+( c*u-a*w+w*x-u*z)*st
    z1 = (c*(u*u+v*v)-w*(a*u+b*v-u*x-v*y-w*z))*(1.0-ct)+z*ct+(-b*u+a*v-v*x+u*y)*st
    return np.array([x1,y1,z1])

def close(nclose, old_xyz, squish=1.0):
    old_ltot,ktot,jtot,nvar = old_xyz.shape
    ltot = old_ltot + nclose

    half = int((jtot+1)/2)-1

    xyz = np.zeros((ltot,ktot,jtot,3))

    xyz[:old_ltot,:,:,:] = old_xyz

    l = old_ltot-1

    # from tail to nose
    axis  = xyz[l, 0, half,:] - xyz[l, 0, 0,:]
    # from up down
    # out   = xyz[l, 0, half, :] - xyz[l-1, 0, half, :]
    out   = xyz[l, 0,    0, :] - xyz[l-1, 0,    0, :]
    out   = out  / np.sqrt(np.dot(out,  out )) #
    down  = np.cross(out, axis)
    chord = np.sqrt(np.dot(axis, axis))        # chord
    axis  = axis / chord
    down  = down / np.sqrt(np.dot(down, down)) #
    
    # EPS = 1.0e-5
    EPS   = 0.5*( xyz[old_ltot-1,0,1] - xyz[old_ltot-1,0,jtot-2] )/(1.5*nclose);
    EPS   = np.sqrt(np.dot(EPS,EPS))
    # EPS   = max(EPS, 5.0e-5)
    # print "EPS is : ", EPS, " Out is ", out

    # all closing planes equal to last plane
    for l in range(old_ltot, ltot):
        for k in range(ktot):
            for j in range(jtot):
                xyz[l,k,j,:]  = xyz[old_ltot-1,k,j,:]

    # tail pts given thickness
    for k in range(ktot):
        for l in range(old_ltot, ltot):
            xyz[l,k,jtot-1,:] = xyz[l-1,k,jtot-1,:] - down*EPS
            xyz[l,k,0,:]      = xyz[l-1,k,0,:] + down*EPS

    for j in range(0, half):
    # for j in range(0, 5):
        j1 = jtot-1-j
        squish_facs = np.linspace(squish, 1.0, ktot)
        for k in range(ktot):
            # point the rotate vector goes through (half way
            a,b,c    = 0.5*(xyz[old_ltot-1, k, j, :]+xyz[old_ltot-1,k,j1,:])
            # a,b,c    = xyz[old_ltot-1, 0, 0, :]
            tmpvec      = xyz[old_ltot-1, k, j, :] - xyz[old_ltot-1,k,j1,:]
            tmpvec      = np.cross(tmpvec, out)
            magvec      = np.sqrt(np.dot(tmpvec, tmpvec))
            #
            if(magvec > 1e-15):
                u,v,w      = tmpvec / magvec
            else:
                u,v,w      = axis
                # u,v,w      = 0.0,0.0,0.0
            for l in range(old_ltot, ltot):
                debug = (k == 1 and j == 56 and l == old_ltot+2)
                # u,v,w = axis
                x,y,z    = xyz[l, k, j, :] # point we will rotate
                x1,y1,z1 = xyz[l, k, j1,:] # point we will rotate
                # if(j==0 and k==0 and l==old_ltot):
                #     print "rotate the point x,y,z about the vector u,v,w that passes through a,b,c"
                #     print "%16.4e %16.4e %16.4e"%(x1,y1,z1)
                #     print "%16.4e %16.4e %16.4e"%(u,v,w)
                #     print "%16.4e %16.4e %16.4e"%(a,b,c)
                # rotate the point x,y,z about the vector u,v,w that passes through a,b,c
                squish_fac = squish_facs[k]
                index = l - (old_ltot-1)
                theta = (90.0*index / nclose)*np.pi/180.0
                if(j == 0 or j == jtot-1):
                    theta = np.pi/2.0
                    # squish_fac = 1.0
                ct = np.cos(theta)
                st = np.sin(theta)

                # lower point ----------------------------------------------------
                delta = rotate(x,y,z,a,b,c,u,v,w,ct,st) - xyz[l,k,j,:]
                if(squish != 1.0):
                    delta = delta - np.dot( delta, out*(1.0-squish_fac) )*out
                # if(abs(np.dot(delta,out)) < (l-old_ltot+1)*EPS):
                #     delta = delta -out*np.dot(delta, out) + (l-old_ltot+1)*EPS*out
                xyz[l,k,j,:] = xyz[l,k,j,:] + delta
                # ----------------------------------------------------------------
                
                # upper point ----------------------------------------------------
                delta = rotate(x1,y1,z1,a,b,c,u,v,w,ct,-st) - xyz[l,k,j1,:]
                if(squish != 1.0):
                    delta = delta - np.dot( delta, out*(1.0-squish_fac) )*out
                # if(abs(np.dot(delta,out)) < (l-old_ltot+1)*EPS):
                #     delta = delta -out*np.dot(delta, out) + (l-old_ltot+1)*EPS*out
                xyz[l,k,j1,:] = xyz[l,k,j1,:] + delta
                # ---------------------------------------------------------------


    # upper and lower surface of collapse is the same
    for j in range(half):
        j1 = jtot-1-j
        for k in range(ktot):
            l = ltot-1
            xyz[l,k,j1] = xyz[l,k,j]

    # #
    # # NOSE
    # # 
    # # take a guess at where the cut at the nose should be
    # for k in range(ktot):
    #     for l in range(old_ltot, ltot):
    #         index = l - (old_ltot-1)
    #         # xyz[l,k,half] = xyz[l,k,half] + np.dot( xyz[l,k,half+1]-xyz[l,k,half], out )
    #         # xyz[l,k,half] = 0.5*(xyz[l,k,half+1] + xyz[l,k,half-1])
    #         xyz[l,k,half] = xyz[l,k,half] + out*EPS*index

    cpy = np.copy(xyz)            
    # smooth in j-direction (last plane only)
    for j in range(half, half+1):
        for k in range(ktot):
            l = ltot-1
            fac = (1.0-1.0/nclose) #0.5 + 0.4*np.sqrt(k*1.0/ktot)
            cpy[l,k,j] = xyz[l,k,j] + fac*(0.5*(xyz[l,k,j+1] + xyz[l,k,j-1])-xyz[l,k,j])
    xyz = np.copy(cpy)

    # smooth in l-direction
    for it in range(nclose*3):
        for j in range(half-2, half+3):
            for k in range(ktot):
                for l in range(old_ltot, ltot-1):
                    cpy[l,k,j] = 0.5*(xyz[l+1,k,j] + xyz[l-1,k,j])
        xyz = np.copy(cpy)

    #
    # TAIL
    #
    # smooth in j-direction (last plane only)
    for j in [-1,0]:
        jp = j+1 if (j!=-1) else  1
        jm = j-1 if (j!= 0) else -2
        # dont do the surface point
        for k in range(0,ktot):
            l = ltot-1
            # Relaxation factor: start half way between tail and next jpoint,
            # increasing up to 0.9 as we increase k
            # fac = 0.0 + 0.8*np.sqrt(k*1.0/ktot)
            fac = (1.0-1.0/nclose)*(0.5 + 0.5 * np.tanh(2*(k*1.0/ktot-0.5)*np.pi))
            cpy[l,k,j] = xyz[l,k,j] + fac*(0.5*(xyz[l,k,jp] + xyz[l,k,jm])-xyz[l,k,j])
    xyz = np.copy(cpy)

    # smooth in l-direction
    for it in range(nclose*3):
        for j in [-3,-2,-1,0,1,2]:
            # dont do the surface point
            for k in range(0,ktot):
                # do the last plane first
                for l in range(old_ltot, ltot-1):
                    cpy[l,k,j] = 0.5*(xyz[l+1,k,j] + xyz[l-1,k,j])
        xyz = np.copy(cpy)
    
    return xyz

    
def close_both(nclose1, nclose2, xyz, squish=1.0):
    print("End Cap...")
    grid1 = close(nclose2, xyz, squish)
    grid1 = grid1[::-1,:,::-1,:]
    print("Start Cap...")
    grid2 = close(nclose1, grid1, squish)
    grid2 = grid2[::-1,:,::-1,:]

    final = np.array(grid2, copy=True, order="C")
    del grid1, grid2, xyz

    # final = np.array(grid1, copy=True, order="C")
    # del grid1, xyz    
    return final
    
