
# import g2d_to_c81
import postprocess_g2d as ppg
import matplotlib.pyplot as plt
from airfoilprep import Airfoil, Polar
import numpy as np

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

all_outputs = ppg.collect_g2d_outputs(".","7a_foil",doplot=False)
for ir in range(len(all_outputs['Re'])):

    table = None
    
    for im in range(len(all_outputs['Mach'])):
        Re    = all_outputs['Re'][ir]
        M     = all_outputs['Mach'][im]
        alpha = all_outputs['alpha']
        cl    = all_outputs['Cl'][:,im,ir]
        cd    = all_outputs['Cd'][:,im,ir]
        cm    = all_outputs['Cm'][:,im,ir]

        # cl = lowpass_filter_fft(alpha, cl, cutoff_ratio=0.1)
        # cd = lowpass_filter_fft(alpha, cd, cutoff_ratio=0.1)
        # cm = lowpass_filter_fft(alpha, cm, cutoff_ratio=0.1)

        p = Polar(Re, alpha, cl, cd, cm)
        af = Airfoil([p]) # can include multiple polars

        cdmax = 2.0

        af2 = af.extrapolate(cdmax, nalpha=11)

        alpha_new, Re_new, cl_new, cd_new, cm_new = af2.createDataGrid()

        if table is None:
            table = np.zeros((len(all_outputs['Mach'])+1, len(alpha_new), 3), 'd')

        # print(alpha_new)

        table[im,:,0] = cl_new[:,0]
        table[im,:,1] = cd_new[:,0]
        table[im,:,2] = cm_new[:,0]

        # print(Re, M)
        # plt.plot(alpha_new, cl_new, '-')
        # plt.plot(alpha, cl, '--')
        plt.plot(alpha_new, cd_new, '-')
        plt.plot(alpha, cd, '--')
        # plt.plot(alpha_new, cm_new, '-')
        # plt.plot(alpha, cm, '--')
        plt.show()
    write_c81("7a_foil.c81", "7afoil", alpha_new, all_outputs['Mach'], table)

        # print(Re_new)
        # print(alpha_new)
        # print(ir, im)
    

# print(all_outputs)
