import numpy as np
import scipy.interpolate



def smooth(xyz):

    ltot,ktot,jtot,var = xyz.shape

    for k in range(1, ktot):
        print("doing k = ", k)
        ratio = (k*1.0/(ktot-1))
        ratio = (10.0 - 15.0*ratio + 6.0*ratio*ratio)*ratio*ratio*ratio

        x = xyz[:,k,:,0]
        y = xyz[:,k,:,1]
        z = xyz[:,k,:,2]
        xd      = np.diff(x, axis=0)
        yd      = np.diff(y, axis=0)
        zd      = np.diff(z, axis=0)
        dist    = np.sqrt(xd**2+yd**2+zd**2)
        u       = np.zeros((ltot,jtot))
        u[1:,:] = np.cumsum(dist, axis=0)
        # u       = np.hstack([[0],u])

        length = u.max(axis=0)
        eq = np.zeros_like(u)
        for j in range(jtot):
            eq[:,j] = np.linspace(0, length[j], ltot)

        new_x = np.copy(x)
        new_y = np.copy(y)
        new_z = np.copy(z)
        # if(
        for j in range(jtot):
            # new_x[1:-1,j] = scipy.interpolate.interp1d(u[1:-1,j],x[1:-1,j])(eq[1:-1,j])
            # new_y[1:-1,j] = scipy.interpolate.interp1d(u[1:-1,j],y[1:-1,j])(eq[1:-1,j])
            # new_z[1:-1,j] = scipy.interpolate.interp1d(u[1:-1,j],z[1:-1,j])(eq[1:-1,j])
            new_x[:,j] = scipy.interpolate.pchip_interpolate(u[:,j],x[:,j],eq[:,j],axis=1)
            new_y[:,j] = scipy.interpolate.pchip_interpolate(u[:,j],y[:,j],eq[:,j],axis=1)
            new_z[:,j] = scipy.interpolate.pchip_interpolate(u[:,j],z[:,j],eq[:,j],axis=1)

        
        xyz[:,k,:,0] = x + (new_x - x)*ratio
        xyz[:,k,:,1] = y + (new_y - y)*ratio
        xyz[:,k,:,2] = z + (new_z - z)*ratio

    return xyz
