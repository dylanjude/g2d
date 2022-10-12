import numpy, os, copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

all_markers = list(Line2D.markers.keys())
field_width = 7
def read_c81_table(f,nMach,nAoA):
    """
    read all mach numbers and data for Cl/Cd/Cm
    Args:
        f: open file handle in read mode
        nMach: number of mach # points
        nAoA: number of AoA points

    Returns:
        data: dict with AoA, Mach and values of coeff in 2d array
    """
# read all mach numbers
    nrows = int(numpy.ceil(nMach/9))
    all_Mach = []
    for irow in range(nrows):
        line = next(f)
        for iMach in range(nMach):
            all_Mach.append(float(line[(iMach+1)*field_width:(iMach+2)*field_width]))

# read AoA and tables
    all_AoA = []
    table   = numpy.zeros((nAoA,nMach))

# loop over AoA, read
    for ialpha in range(nAoA):
        line    = next(f)
# remember AoA
        all_AoA.append(float(line[0:field_width]))
# append other mach numbers in continuing rows to original list
        for irow in range(1,nrows):
            temp = next(f)
            line += temp
# now we can slot in table values 
        for iMach in range(nMach):
            table[ialpha,iMach] = line[(iMach+1)*field_width:(iMach+2)*field_width]

# add all values of Alpha, Mach and table to data dict
    data = {'AoA': numpy.asarray(all_AoA), 'Mach': numpy.asarray(all_Mach), 'table': table}
    return data

def read_c81_file(filename):
    """
    read dictionary data to c81 format file

    Args:
        filename: c81 filename

    Returns: 
        data: dictionary with alpha, Mach, Cl, Cd, Cm
    """
    with open(filename,'r') as f:
        line         = next(f)
        airfoil_name = line[:30]
        nCL_Mach     = int(line[30:32])
        nCL_AoA      = int(line[32:34])
        nCD_Mach     = int(line[34:36])
        nCD_AoA      = int(line[36:38])
        nCM_Mach     = int(line[38:40])
        nCM_AoA      = int(line[40:42])
        Cl_data      = read_c81_table(f,nCL_Mach,nCL_AoA)
        Cd_data      = read_c81_table(f,nCD_Mach,nCD_AoA)
        Cm_data      = read_c81_table(f,nCM_Mach,nCM_AoA)

        all_data = {'Cl_data': Cl_data, 'Cd_data': Cd_data, 'Cm_data': Cm_data}
    return all_data

def plot_c81(data_dict, airfoil_name, separate_plots=False):

    """
    plot Cl vs alpha, Cd vs alpha and Cm vs alpha for each Mach
    """
    Cl_data = data_dict['Cl_data']
    Cd_data = data_dict['Cd_data']
    Cm_data = data_dict['Cm_data']
    plt.figure(1)
    ax1  = plt.gca()
    if separate_plots:
        plt.figure(2)
        ax2 = plt.gca()
        plt.figure(3)
        ax3 = plt.gca()
    else:
        ax2  = ax1.twinx()
        ax3  = ax1.twinx()
        ax3.spines['right'].set_position(("axes", 1.2))    
    for imach,Mach in enumerate(Cl_data['Mach']):
        ax1.plot(Cl_data['AoA'],Cl_data['table'][:,imach],marker=all_markers[imach],label='Cl: Mach={:}'.format(Mach),color='C0')
    for imach,Mach in enumerate(Cd_data['Mach']):
        ax2.plot(Cd_data['AoA'],Cd_data['table'][:,imach],marker=all_markers[imach],label='Cd: Mach={:}'.format(Mach),color='C1')
    for imach,Mach in enumerate(Cm_data['Mach']):
        ax3.plot(Cm_data['AoA'],Cm_data['table'][:,imach],marker=all_markers[imach],label='Cm: Mach={:}'.format(Mach),color='C2')

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

    if separate_plots:
        ax1.set_title('Lift coeff. vs. AoA at various Mach')
        plt.figure(1)
        plt.grid(True,linestyle='--',alpha=0.3)
        plt.tight_layout()
        ax1.legend(loc='best')
        filename = '{:s}_Cl.png'.format(airfoil_name)
        plt.savefig(filename,dpi=300)
        print('saved a file called {:s}'.format(filename))
        plt.close()

        ax2.set_title('Drag coeff. vs. AoA at various Mach')
        plt.figure(2)
        plt.grid(True,linestyle='--',alpha=0.3)
        plt.tight_layout()
        ax2.legend(loc='best')
        filename = '{:s}_Cd.png'.format(airfoil_name)
        plt.savefig(filename,dpi=300)
        print('saved a file called {:s}'.format(filename))
        plt.close()

        ax3.set_title('Pitching mom. coeff. vs. AoA at various Mach')
        plt.figure(3)
        plt.grid(True,linestyle='--',alpha=0.3)
        plt.tight_layout()
        ax3.legend(loc='best')
        filename = '{:s}_Cm.png'.format(airfoil_name)
        plt.savefig(filename,dpi=300)
        print('saved a file called {:s}'.format(filename))
        plt.close()

    else:
        plt.title('Cl, Cd, Cm vs. AoA at various Mach')
        plt.grid(True,linestyle='--',alpha=0.3)
        ax1.legend(loc='best')
        plt.tight_layout()
        filename = '{:s}_ClCdCm.png'.format(airfoil_name)
        plt.savefig(filename,dpi=300)
        print('saved a file called {:s}'.format(filename))
        plt.close()

    return None

def overlay_many_c81(c81_dict, prefix):

    """
    plot Cl vs alpha, Cd vs alpha and Cm vs alpha for many airfoils

    Each airfoil has a different color
    Each mach has a different marker

    """

    colors   = list(mcolors.TABLEAU_COLORS.keys())
    nsets    = len(c81_dict)
    plt.figure(1)
    ax1     = plt.gca()
    plt.figure(2)
    ax2     = plt.gca()
    plt.figure(3)
    ax3     = plt.gca()
    plt.figure(4)
    ax4     = plt.gca()


# add insert showing Cd near zero alpha
    axins = inset_axes(ax2, width="80%", height="100%",
                        bbox_to_anchor=(0.1, .4, .6, .4),
                        bbox_transform=ax2.transAxes, loc=8)    
    for iset,(label, data_dict) in enumerate(c81_dict.items()):
        Cl_data = data_dict['Cl_data']
        Cd_data = data_dict['Cd_data']
        Cm_data = data_dict['Cm_data']

        marker  = all_markers[iset]
        color   = colors[iset]
# plot Cl vs Aoa for various Mach for this airfoil    
        plot_Mach_sweep(ax1, Cl_data, label, marker, color)

        plot_Mach_sweep(ax2, Cd_data, label, marker, color)

        plot_Mach_sweep(ax3, Cm_data, label, marker, color)

        plot_Cd_Cl(ax4, Cd_data, Cl_data, label, marker, color)

    ax1.set_ylabel('Cl'); ax1.set_xlabel('AoA (deg)')
    ax2.set_ylabel('Cd'); ax2.set_xlabel('AoA (deg)')
    ax3.set_ylabel('Cm'); ax3.set_xlabel('AoA (deg)')
    ax4.set_ylabel('Cl'); ax4.set_xlabel('Cd')

    ax1.set_title('Lift coeff. vs. AoA at various Mach')
    plt.figure(1)
    plt.grid(True,linestyle='--',alpha=0.3)
    plt.tight_layout()
    ax1.legend(loc='best')
    filename = os.path.join(prefix,'all_Cl.png')
    plt.savefig(filename,dpi=300)
    print('saved a file called {:s}'.format(filename))
    plt.close()

    ax2.set_title('Drag coeff. vs. AoA at various Mach')
    plt.figure(2)
    ax2.grid(True,linestyle='--',alpha=0.3)
    plt.tight_layout()
    ax2.legend(loc='best')

# find alpha where Cd < 0.01
    right_end = 180
    box_Cd_top= 0.015
    for iset,(label, data_dict) in enumerate(c81_dict.items()):
        Cd_data = data_dict['Cd_data']
        marker  = all_markers[iset]
        color   = colors[iset]
        ncols  = Cd_data['table'].shape[1]
        max_id = 1e6
        for col in range(ncols):
            col_vals = Cd_data['table'][:,col]
            ialpha   = numpy.asarray(col_vals < box_Cd_top).nonzero()[0]
            if ialpha[-1] < max_id:
                max_id = ialpha[-1]

        if Cd_data['AoA'][max_id] < right_end:
            right_end = Cd_data['AoA'][max_id]

        print(Cd_data['AoA'][max_id])
        copy_Cd_data          = copy.deepcopy(Cd_data)
        copy_Cd_data['AoA']   = copy_Cd_data['AoA'][:max_id]
        copy_Cd_data['table'] = numpy.zeros((max_id,ncols))
        for col in range(ncols):
            copy_Cd_data['table'][:,col] = Cd_data['table'][:max_id,col]
        plot_Mach_sweep(axins, copy_Cd_data, label, marker, color)

    axins.set_ylim(bottom=0,top=box_Cd_top)
    axins.set_xlim(right=right_end)
    axins.grid(True,alpha=0.3,linestyle='--')
    axins.spines['bottom'].set_color('0.7')
    axins.spines['left'].set_color('0.7')
    axins.spines['right'].set_color('0.7')
    axins.spines['top'].set_color('0.7')
    axins.xaxis.label.set_color('0.7')
    axins.yaxis.label.set_color('0.7')
    axins.tick_params(axis='x', colors='0.7')    
    axins.tick_params(axis='y', colors='0.7')    
    filename = os.path.join(prefix,'all_Cd.png')
    ax2.set_ylim(bottom=0)

    plt.savefig(filename,dpi=300)
    print('saved a file called {:s}'.format(filename))
    plt.close()

    ax3.set_title('Pitching mom. coeff. vs. AoA at various Mach')
    plt.figure(3)
    plt.grid(True,linestyle='--',alpha=0.3)
    plt.tight_layout()
    ax3.legend(loc='best')
    filename = os.path.join(prefix,'all_Cm.png')
    plt.savefig(filename,dpi=300)
    print('saved a file called {:s}'.format(filename))
    plt.close()

    ax4.set_title('Cd vs. Cl at various Mach')
    plt.figure(4)
    plt.grid(True,linestyle='--',alpha=0.3)
    plt.tight_layout()
    ax4.legend(loc='best')
    filename = os.path.join(prefix,'all_Cd_vs_Cl.png')
    plt.savefig(filename,dpi=300)
    print('saved a file called {:s}'.format(filename))
    plt.close()

    return None

def plot_Mach_sweep(ax, data, label, marker, color):
    nMach    = len(data['Mach'])
    if nMach == 1:
        dalpha = 1.0
        alpha_min = 1.0
    else:
        alpha_min= 0.2
        alpha_max= 1.0
        dalpha   = (alpha_max-alpha_min)/float(nMach-1)
    for imach,Mach in enumerate(data['Mach']):
        alpha   = alpha_min + dalpha*float(imach)
        l_label = '{:s} @ Mach={:}'.format(label,Mach)
        ax.plot(data['AoA'],data['table'][:,imach],marker=marker, \
                label=l_label,color=color, alpha=alpha)
    return None

def plot_Cd_Cl(ax, Cd, Cl, label, marker, color):
    nMach    = len(Cd['Mach'])

# set transparency
    if nMach == 1:
        dalpha = 1.0
        alpha_min = 1.0
    else:
        alpha_min= 0.2
        alpha_max= 1.0
        dalpha   = (alpha_max-alpha_min)/float(nMach-1)

# loop over Mach
    for imach,Mach in enumerate(Cd['Mach']):
        alpha   = alpha_min + dalpha*float(imach)
        l_label = '{:}: Cd vs Cl @ Mach={:}'.format(label,Mach)
        ax.plot(Cd['table'][:,imach],Cl['table'][:,imach],marker=marker, \
                label=l_label,color=color, alpha=alpha)
    return None