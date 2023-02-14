import numpy, os, sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
all_markers = list(Line2D.markers.keys())

def read_g2d_output(prefix,R,aoa,M,plot_history,av=200):
    """
    read g2d output file for a single run case

    Args:
        prefix: string prepended to filename pattern 
        R     : Reynolds number
        aoa   : angle of attack, deg 
        M     : Mach number
        plot_history: boolean; True = plot time history of Cl, Cd, Cm for the case
    """

# identify file to read
    filename = "{:s}_r{:7.0f}_a{:05.1f}_m{:04.2f}.forces".format(prefix,R,aoa%360,M)

# load the file
    cl,cd,cm = numpy.loadtxt(filename,comments=('#')).T

# G2D converged in steady mode if < 1000 time steps
    if(cl.size < 999):
        # print('pt')
        Cl,Cd,Cm = cl[-1],cd[-1],cm[-1]

# unsteady mode starts: average forces
    else:
        # print('av')
        Cl,Cd,Cm = numpy.average(cl[-av:]),numpy.average(cd[-av:]),numpy.average(cm[-av:])

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
        plt.savefig(filename.replace('.forces','.png'))
        plt.close()
    return Cl,Cd,Cm

foil_name = sys.argv[1]

# collect all outputs
all_outputs = []
for root, dirs, files in os.walk(foil_name):
    for file in files:
        if file.endswith('.forces'):
            all_outputs.append(os.path.join(root,file))

prefix    = foil_name + os.sep + foil_name
# collect all Reynolds/Mach numbers/angles of attack
all_Re    = []
all_Mach  = []
all_alpha = []
for output in all_outputs:
    temp = output.replace(prefix,'').replace('.forces','').split('_')[1:]
    Re   = float(temp[0][1:])
    if Re not in all_Re:
        all_Re.append(Re)
    Mach = float(temp[2][1:])
    if Mach not in all_Mach:
        all_Mach.append(Mach)
    alpha = float(temp[1][1:])
    if alpha > 180:
        alpha = alpha - 360
    if alpha not in all_alpha:
        all_alpha.append(alpha)

all_alpha = numpy.asarray(all_alpha); all_alpha.sort()
all_Re    = numpy.asarray(all_Re); all_Re.sort()
all_Mach  = numpy.asarray(all_Mach); all_Mach.sort()



Cl     = numpy.zeros((len(all_alpha),len(all_Mach),len(all_Re)))
Cd     = numpy.zeros_like(Cl)
Cm     = numpy.zeros_like(Cl)

for ire,Re in enumerate(all_Re):
    for imach,Mach in enumerate(all_Mach):
        for ialpha,alpha in enumerate(all_alpha):
            Cl[ialpha,imach,ire], Cd[ialpha,imach,ire], Cm[ialpha,imach,ire] = \
                read_g2d_output(prefix,Re,alpha,Mach,plot_history=True)


# plot Cl vs alpha, Cd vs alpha and Cm vs alpha for each Mach at various Re
for ire,Re in enumerate(all_Re):
    plt.figure(1)
    ax1  = plt.gca()
    ax2  = ax1.twinx()
    ax3  = ax1.twinx()
    ax3.spines['right'].set_position(("axes", 1.2))    
    for imach,Mach in enumerate(all_Mach):
        ax1.plot(all_alpha,Cl[:,imach,ire],marker=all_markers[imach],label='Cl: Mach={:}'.format(Mach),color='C0')
        ax2.plot(all_alpha,Cd[:,imach,ire],marker=all_markers[imach],label='Cd: Mach={:}'.format(Mach),color='C1')
        ax3.plot(all_alpha,Cm[:,imach,ire],marker=all_markers[imach],label='Cm: Mach={:}'.format(Mach),color='C2')

    ax1.set_ylabel('Cl')
    ax2.set_ylabel('Cd')
    ax3.set_ylabel('Cm')
    ax1.yaxis.label.set_color('C0')
    ax2.yaxis.label.set_color('C1')
    ax3.yaxis.label.set_color('C2')        

    tkw = dict(size=4, width=1.5)
    ax1.tick_params(axis='y', colors='C0', **tkw)
    ax2.tick_params(axis='y', colors='C1', **tkw)
    ax3.tick_params(axis='y', colors='C2', **tkw)

    plt.title('Reynolds number = {:}'.format(Re))
    plt.grid(True,linestyle='--',alpha=0.3)
    ax1.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.savefig('ClCdCm_vs_Mach_Re_{:}.png'.format(Re),dpi=300)
    plt.close()

# plot Cl vs alpha, Cd vs alpha and Cm vs alpha for each Re at various Mach
