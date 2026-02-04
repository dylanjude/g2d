#!/usr/bin/env python3
"""
Plot Cl, Cd, and Cm vs angle-of-attack for two C81 airfoil files.

Usage:
    python plot_c81.py file1.c81 file2.c81 [--mach MACH_INDEX]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_c81(filename):
    """
    Parse a C81 airfoil data file.
    
    C81 format (fixed-width columns, values can run together with negative signs):
    - Line 1: Airfoil name (30 chars) + NL ND NM (number of entries per table, encoded)
    - For each table (CL, CD, CM):
        - Mach number header line
        - Alpha and coefficient data rows (alpha + one coef per Mach)
    
    Returns dict with keys: 'name', 'mach_cl', 'mach_cd', 'mach_cm', 
                           'alpha_cl', 'alpha_cd', 'alpha_cm', 'cl', 'cd', 'cm'
    where cl, cd, cm are 2D arrays [mach_index, alpha_index]
    """
    with open(filename, 'r') as f:
        lines = [line.rstrip('\n\r') for line in f.readlines()]
    
    # First line: airfoil name (first 30 chars) + counts
    # The counts are n_alpha * n_mach for each table
    header = lines[0]
    name = header[:30].strip()
    
    # Parse the rest of header for counts (may help validate but we'll auto-detect)
    line_idx = 1
    
    def parse_data_line(line):
        """
        Parse a fixed-width C81 data line where values can run together.
        Format: 7 characters for alpha, then 7 chars per coefficient.
        Negative signs can consume the space between values.
        Numbers may start with decimal point (e.g., .35 instead of 0.35)
        """
        import re
        # Match numbers including:
        # - Optional negative sign
        # - Optional digits before decimal
        # - Optional decimal point
        # - Optional digits after decimal
        # Must have at least one digit somewhere
        pattern = r'-?(?:\d+\.?\d*|\.\d+)'
        matches = re.findall(pattern, line)
        return [float(m) for m in matches]
    
    def read_table(start_idx, expected_n_mach=None):
        """Read a coefficient table (CL, CD, or CM)."""
        idx = start_idx
        
        # First line should be Mach numbers - they're indented with leading spaces
        # and contain only small positive values (typically 0 < M < 2)
        line = lines[idx]
        mach_values = parse_data_line(line)
        
        # Validate this looks like a Mach header:
        # - Starts with whitespace (indented)
        # - All values are non-negative and reasonable for Mach numbers
        if not (line.startswith(' ') and all(0 <= v < 3.0 for v in mach_values)):
            raise ValueError(f"Expected Mach header at line {idx+1}: {line}")
        
        mach_numbers = mach_values
        n_mach = len(mach_numbers)
        idx += 1
        
        # Read data rows until we hit another Mach header or EOF
        alpha_values = []
        coef_data = []
        
        while idx < len(lines):
            line = lines[idx]
            if not line.strip():
                idx += 1
                continue
            
            values = parse_data_line(line)
            
            # Data rows have n_mach + 1 values (alpha + coefficients)
            # Mach headers have exactly n_mach values and start with whitespace
            if len(values) == n_mach + 1:
                alpha_values.append(values[0])
                coef_data.append(values[1:])
                idx += 1
            elif len(values) == n_mach and line.startswith(' ') and all(0 <= v < 3.0 for v in values):
                # This is the next Mach header - stop here
                break
            else:
                # Unexpected format - try to continue
                idx += 1
        
        alpha = np.array(alpha_values)
        coefs = np.array(coef_data).T  # Shape: (n_mach, n_alpha)
        
        return idx, np.array(mach_numbers), alpha, coefs
    
    # Read CL table
    line_idx, mach_cl, alpha_cl, cl_data = read_table(line_idx)
    
    # Read CD table
    line_idx, mach_cd, alpha_cd, cd_data = read_table(line_idx)
    
    # Read CM table
    line_idx, mach_cm, alpha_cm, cm_data = read_table(line_idx)
    
    return {
        'name': name,
        'mach_cl': mach_cl,
        'mach_cd': mach_cd,
        'mach_cm': mach_cm,
        'alpha_cl': alpha_cl,
        'alpha_cd': alpha_cd,
        'alpha_cm': alpha_cm,
        'cl': cl_data,
        'cd': cd_data,
        'cm': cm_data
    }
    
    # Read CL table
    line_idx, mach_cl, alpha_cl, cl_data = read_table(line_idx, n_mach_cl)
    
    # Read CD table
    line_idx, mach_cd, alpha_cd, cd_data = read_table(line_idx, n_mach_cd)
    
    # Read CM table
    line_idx, mach_cm, alpha_cm, cm_data = read_table(line_idx, n_mach_cm)
    
    return {
        'name': name,
        'mach_cl': np.array(mach_cl),
        'mach_cd': np.array(mach_cd),
        'mach_cm': np.array(mach_cm),
        'alpha_cl': alpha_cl,
        'alpha_cd': alpha_cd,
        'alpha_cm': alpha_cm,
        'cl': cl_data,
        'cd': cd_data,
        'cm': cm_data
    }


def interpolate_to_mach(data, target_mach, coef_key, mach_key, alpha_key):
    """
    Linearly interpolate coefficient data to a target Mach number.
    
    Parameters:
        data: parsed C81 data dict
        target_mach: desired Mach number
        coef_key: 'cl', 'cd', or 'cm'
        mach_key: 'mach_cl', 'mach_cd', or 'mach_cm'
        alpha_key: 'alpha_cl', 'alpha_cd', or 'alpha_cm'
    
    Returns:
        alpha, coef arrays at the interpolated Mach number
    """
    mach_arr = data[mach_key]
    coef_arr = data[coef_key]  # Shape: (n_mach, n_alpha)
    alpha = data[alpha_key]
    
    # Check bounds
    if target_mach <= mach_arr[0]:
        return alpha, coef_arr[0, :]
    if target_mach >= mach_arr[-1]:
        return alpha, coef_arr[-1, :]
    
    # Find bracketing indices
    idx_upper = np.searchsorted(mach_arr, target_mach)
    idx_lower = idx_upper - 1
    
    # Linear interpolation weight
    m_lo, m_hi = mach_arr[idx_lower], mach_arr[idx_upper]
    t = (target_mach - m_lo) / (m_hi - m_lo)
    
    coef_interp = (1 - t) * coef_arr[idx_lower, :] + t * coef_arr[idx_upper, :]
    
    return alpha, coef_interp


def plot_c81_comparison(file1, file2, mach=0.0, output=None):
    """
    Plot Cl, Cd, Cm vs AoA for two C81 files at a specified Mach number.
    
    Parameters:
        file1, file2: Paths to C81 files
        mach: Mach number to plot (will interpolate if not exact match)
        output: Output filename (optional, displays if None)
    """
    data1 = parse_c81(file1)
    data2 = parse_c81(file2)
    
    label1 = Path(file1).stem
    label2 = Path(file2).stem
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # CL plot
    ax = axes[0]
    alpha1, cl1 = interpolate_to_mach(data1, mach, 'cl', 'mach_cl', 'alpha_cl')
    alpha2, cl2 = interpolate_to_mach(data2, mach, 'cl', 'mach_cl', 'alpha_cl')
    i1 = np.where(np.abs(alpha1)<20)
    i2 = np.where(np.abs(alpha2)<20)
    alpha1,cl1 = alpha1[i1],cl1[i1]
    alpha2,cl2 = alpha2[i2],cl2[i2]
    ax.plot(alpha1, cl1, 'b-o', label=f'{label1}', markersize=4, markerfacecolor='none')
    ax.plot(alpha2, cl2, 'r-s', label=f'{label2}', markersize=4, markerfacecolor='none')
    ax.set_xlabel('Angle of Attack (deg)')
    ax.set_ylabel('$C_l$')
    ax.set_title(f'Lift Coefficient (M={mach:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    # ax.set_xlim(-20, 20)
    
    # CD plot
    ax = axes[1]
    alpha1, cd1 = interpolate_to_mach(data1, mach, 'cd', 'mach_cd', 'alpha_cd')
    alpha2, cd2 = interpolate_to_mach(data2, mach, 'cd', 'mach_cd', 'alpha_cd')
    i1 = np.where(np.abs(alpha1)<20)
    i2 = np.where(np.abs(alpha2)<20)
    alpha1,cd1 = alpha1[i1],cd1[i1]
    alpha2,cd2 = alpha2[i2],cd2[i2]
    ax.plot(alpha1, cd1, 'b-o', label=f'{label1}', markersize=4, markerfacecolor='none')
    ax.plot(alpha2, cd2, 'r-s', label=f'{label2}', markersize=4, markerfacecolor='none')
    ax.set_xlabel('Angle of Attack (deg)')
    ax.set_ylabel('$C_d$')
    ax.set_title(f'Drag Coefficient (M={mach:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    # ax.set_xlim(-20, 20)
    
    # CM plot
    ax = axes[2]
    alpha1, cm1 = interpolate_to_mach(data1, mach, 'cm', 'mach_cm', 'alpha_cm')
    alpha2, cm2 = interpolate_to_mach(data2, mach, 'cm', 'mach_cm', 'alpha_cm')
    i1 = np.where(np.abs(alpha1)<20)
    i2 = np.where(np.abs(alpha2)<20)
    alpha1,cm1 = alpha1[i1],cm1[i1]
    alpha2,cm2 = alpha2[i2],cm2[i2]
    ax.plot(alpha1, cm1, 'b-o', label=f'{label1}', markersize=4, markerfacecolor='none')
    ax.plot(alpha2, cm2, 'r-s', label=f'{label2}', markersize=4, markerfacecolor='none')
    ax.set_xlabel('Angle of Attack (deg)')
    ax.set_ylabel('$C_m$')
    ax.set_title(f'Pitching Moment Coefficient (M={mach:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    # ax.set_xlim(-20, 20)
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output}")
    else:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot Cl, Cd, Cm vs AoA for two C81 airfoil files'
    )
    parser.add_argument('file1', help='First C81 file')
    parser.add_argument('file2', help='Second C81 file')
    parser.add_argument('--mach', '-m', type=float, default=0.0,
                        help='Mach number to plot (default: 0.0, interpolates if needed)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename (displays plot if not specified)')
    
    args = parser.parse_args()
    
    plot_c81_comparison(args.file1, args.file2, mach=args.mach, output=args.output)


if __name__ == '__main__':
    main()
