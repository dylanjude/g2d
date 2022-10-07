import numpy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
all_markers = list(Line2D.markers.keys())

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
        line = next(f).split()
        for mach in line:
            all_Mach.append(float(mach))

# read AoA and tables
    all_AoA = []
    table   = numpy.zeros((nAoA,nMach))

# loop over AoA, read
    for ialpha in range(nAoA):
        line    = next(f).split()
# remember AoA
        all_AoA.append(float(line.pop(0)))
# append other mach numbers in continuing rows to original list
        for irow in range(1,nrows):
            temp = next(f).split()
            line.extend(temp)
# now we can slot in table values 
        for iMach,entry in enumerate(line):
            table[ialpha,iMach] = entry

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