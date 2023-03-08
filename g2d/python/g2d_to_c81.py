def write_c81_table(data,key,f):
    """
    write all mach numbers and data for Cl/Cd/Cm
    Args:
        data: dict with alpha, Mach, Cl, Cd, Cm
        key : which field to write
    """
    f.write('       ')
# write Mach # (assume iRe = 0) 
    for iMach,Mach in enumerate(data['Mach']):
        if iMach%9 ==0:
            f.write('\n       ')
        f.write('{:7.3f}'.format(Mach))

    if key.lower().startswith(('cl','lift','cm','moment')):
        fstr = '{:7.4f}'
    else:
        fstr = '{:7.5f}'        # drag is never negative
# loop over AoA, write
    for ialpha,alpha in enumerate(data['alpha']):
        f.write('\n{:<7.2f}'.format(alpha))
# write data field for various mach #s in this row
        for iMach,Mach in enumerate(data['Mach']):
            if iMach > 2 and iMach%9 ==0:
                f.write('\n       ')
            val = data[key][ialpha,iMach,0]
            # if val < 0:
            # else:
            #     fstr = '{:7.5f}'
            f.write(fstr.format(val))

    return None

def write_c81_file(data, filename):
    """
    write dictionary data to c81 format file

    Args:
        data    : dictionary with alpha, Re, Mach, Cl, Cd, Cm and airfoil name
        filename: c81 filename

    Returns: None
    """
    print('writing a c81 airfoil table file called ',filename)
    with open(filename,'w') as f:
        f.write('{:30s}{:2d}{:2d}{:2d}{:2d}{:2d}{:2d}'.format('G2D output airfoil',len(data['Mach']), len(data['alpha']),
                                                                                   len(data['Mach']), len(data['alpha']),
                                                                                   len(data['Mach']), len(data['alpha'])))

        write_c81_table(data,'Cl', f)
        write_c81_table(data,'Cd', f)
        write_c81_table(data,'Cm', f)

    return None