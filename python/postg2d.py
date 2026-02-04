import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pickle
import glob
import re
from airfoilprep import Airfoil, Polar

def read_g2d_output(filename,R,aoa,M,plot_history,av=200):
    """
    read g2d output file for a single run case

    Args:
        filename: the filename
        R     : Reynolds number
        aoa   : angle of attack, deg 
        M     : Mach number
        plot_history: boolean; True = plot time history of Cl, Cd, Cm for the case
    """

# load the file
    cl,cd,cm = np.loadtxt(filename,comments=('#')).T

# G2D converged in steady mode if < 1000 time steps
    if(cl.size < 999):
        # print('pt')
        Cl,Cd,Cm = cl[-1],cd[-1],cm[-1]

# unsteady mode starts: average forces
    else:
        # print('av')
        Cl,Cd,Cm = np.average(cl[-av:]),np.average(cd[-av:]),np.average(cm[-av:])

# plot history if required
    if plot_history:
        plt.figure(1)
        fig, axs = plt.subplots(nrows=3,sharex=True)        
        axs[0].plot(cl); axs[0].set_ylabel('Cl')
        axs[0].set_title('AoA = {:f} deg'.format(aoa))
        axs[1].plot(cd); axs[1].set_ylabel('Cd')
        axs[2].plot(cm); axs[2].set_ylabel('Cm')
        axs[0].grid(True,linestyle='--',alpha=0.3)
        axs[1].grid(True,linestyle='--',alpha=0.3)
        axs[2].grid(True,linestyle='--',alpha=0.3)
        png_name = filename.replace('.forces','.png')
        plt.savefig(png_name)
        print('saved a file called {:s}'.format(png_name))
        plt.close()
    return Cl,Cd,Cm

def collect_g2d_outputs(directory,doplot=True):
# collect all outputs

    # Get the list of matching files
    all_outputs = glob.glob(os.path.join(directory, "*.forces"))

    foilname = None

    pattern = r'(.+?)_r(\d+)_a([\d.]+)_m([\d.]+)'
    
# collect all Reynolds/Mach numbers/angles of attack
    all_Re    = []
    all_Mach  = []
    all_alpha = []
    for output in all_outputs:
        m = re.search(pattern, os.path.basename(output)[:-7])
        if(not m):
            print("Could not regex-parse filename: ", output)
            continue

        foilname = m.group(1)     
        Re       = float(m.group(2))
        alpha    = float(m.group(3))
        Mach     = float(m.group(4))
        
        if Re not in all_Re:
            all_Re.append(Re)
        if Mach not in all_Mach:
            all_Mach.append(Mach)
        if alpha > 180:
            alpha = alpha - 360
        if alpha not in all_alpha:
            all_alpha.append(alpha)

    all_alpha = np.asarray(all_alpha); all_alpha.sort()
    all_Re    = np.asarray(all_Re);    all_Re.sort()
    all_Mach  = np.asarray(all_Mach);  all_Mach.sort()

    Cl     = np.zeros((len(all_alpha),len(all_Mach),len(all_Re)))
    Cd     = np.zeros_like(Cl)
    Cm     = np.zeros_like(Cl)

    for ire,Re in enumerate(all_Re):
        for imach,Mach in enumerate(all_Mach):
            for ialpha,alpha in enumerate(all_alpha):

                filename = "{:s}_r{:07.0f}_a{:05.1f}_m{:04.2f}.forces".format(foilname,Re,
                                                                              alpha%360,Mach)
                filename = os.path.join(directory,filename)
                
                Cl[ialpha,imach,ire], Cd[ialpha,imach,ire], Cm[ialpha,imach,ire] = \
                    read_g2d_output(filename,Re,alpha,Mach,plot_history=doplot)

    all_outputs = {'alpha': all_alpha, 'Re': all_Re, 'Mach': all_Mach, 
                   'Cl': Cl, 'Cd': Cd, 'Cm': Cm}

    return all_outputs, foilname

def write_c81(fout, foilname, alphas, machs, table):
    out = open(fout, "w")                  
    out.write("{:30s}{:2d}{:2d}{:2d}{:2d}{:2d}{:2d}\n".format(foilname,
                                                              len(machs),len(alphas),
                                                              len(machs),len(alphas),
                                                              len(machs),len(alphas)))
    for ifield in range(3):
        out.write("{:7s}".format(" "))
        for i in range(len(machs)):
            out.write("{:7.4f}".format(machs[i]))
            if((i+1)%9==0):
                out.write("\n{:7s}".format(" "))
        out.write("\n")
        
        for ia in range(len(alphas)):
            out.write("{:7.2f}".format(alphas[ia]))
            for i in range(len(machs)):
                out.write("{:7.4f}".format(table[i,ia,ifield]))
                if((i+1)%9==0):
                    out.write("\n{:7s}".format(" "))
            out.write("\n")
    out.close()

def outputs_to_c81(directory,doplot=False):
    all_outputs,foilname = collect_g2d_outputs(directory,doplot=False)

    # print(all_outputs)

    for ir in range(len(all_outputs['Re'])):
        table = None
        for im in range(len(all_outputs['Mach'])):
            Re    = all_outputs['Re'][ir]
            M     = all_outputs['Mach'][im]
            alpha = all_outputs['alpha']
            cl    = all_outputs['Cl'][:,im,ir]
            cd    = all_outputs['Cd'][:,im,ir]
            cm    = all_outputs['Cm'][:,im,ir]

            p     = Polar(Re, alpha, cl, cd, cm)
            af    = Airfoil([p]) # can include multiple polars
    
            cdmax = 2.0
    
            af2 = af.extrapolate(cdmax, nalpha=11)
    
            alpha_new, Re_new, cl_new, cd_new, cm_new = af2.createDataGrid()
    
            if table is None:
                table = np.zeros((len(all_outputs['Mach'])+1, len(alpha_new), 3), 'd')
    
            table[im,:,0] = cl_new[:,0]
            table[im,:,1] = cd_new[:,0]
            table[im,:,2] = cm_new[:,0]
    
            # # print(Re, M)
            # plt.plot(alpha_new, cl_new, '-')
            # plt.plot(alpha, cl, '--')
            # # plt.plot(alpha_new, cd_new, '-')
            # # plt.plot(alpha, cd, '--')
            # # plt.plot(alpha_new, cm_new, '-')
            # # plt.plot(alpha, cm, '--')
            # plt.show()
        fout = "{:s}_re{:d}.c81".format(foilname,ir)
        print("writing to: ", fout)
        write_c81(fout, foilname, alpha_new, all_outputs['Mach'], table)
        

def pickle_airfoil(directory, airfoil_name):
    all_outputs = collect_g2d_outputs(directory,airfoil_name)
    with open("{:s}.pk".format(airfoil_name), 'wb') as f:
        pickle.dump(all_outputs, f)

if __name__ == "__main__":
    if(len(sys.argv)>1):
        odir = sys.argv[1]
    else:
        quit("Usage: python postg2d.py /path/to/outputs")
    
    outputs_to_c81(odir,False)


# # write c81file from g2d outputs
#     g2d_to_c81.write_c81_file(all_outputs,'{:s}.c81'.format(airfoil_name))

#     return None
