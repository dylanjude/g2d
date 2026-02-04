# g2d

A GPU-accelerated 2D CFD solver and mesh generator for aerodynamic analysis of airfoil sections. Solves the Euler, laminar Navier-Stokes, or RANS equations (with Spalart-Allmaras turbulence model) on structured O-meshes using CUDA.

## Features

- **GPU-accelerated flow solver** with Roe approximate Riemann scheme, diagonalized ADI time-stepping, and GMRES linear solver
- **Selectable equation sets**: Euler, laminar Navier-Stokes, RANS with Spalart-Allmaras
- **Multi-condition sweeps**: batch-solve across arrays of Mach numbers, Reynolds numbers, and angles of attack
- **Structured mesh generation**: O-topology grid generation around arbitrary airfoil geometries with Poisson-based smoothing
- **Python bindings** via pybind11 for both the solver (`garfoil`) and mesh generator (`ogen`)
- **Airfoil utilities**: NACA 4/5-digit generation, Hicks-Henne shape perturbations, trailing edge closure, point distribution tools

## Requirements

- CMake 3.17+
- CUDA toolkit (sm_60 / sm_70 / sm_80 / sm_90)
- C++ compiler with C++11 support
- Python 3 with development headers
- pybind11

## Building

```bash
mkdir build && cd build
cmake ..
make
```

This produces:
- `build/lib/g2d.exe` -- standalone solver executable
- `build/lib/garfoil.so` -- Python solver module
- `build/lib/ogen.so` -- Python mesh generation module

## Usage

### Standalone solver

The executable reads an `inputs.g2d` file from the current working directory:

```
machs   : 0.3 0.5 0.7
aoas    : -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16
reys    : 2500000
airfoil : 7a_foil.x
eqns    : sa
order   : 3
varyRe  : 1
```

| Key       | Description                                          | Default       |
|-----------|------------------------------------------------------|---------------|
| `machs`   | Space-separated Mach numbers                         | `0.3`         |
| `aoas`    | Space-separated angles of attack (degrees)           | `2 4 6`       |
| `reys`    | Space-separated Reynolds numbers                     | `1000000`     |
| `airfoil` | Path to grid file                                    | `naca0012.xyz`|
| `eqns`    | Equation set: `euler`, `laminar`, or `sa`            | `sa`          |
| `order`   | Spatial reconstruction order (2, 3, or 5)            | `3`           |
| `varyRe`  | Scale Reynolds number with Mach (`1` or `0`)         | `1`           |

Run from a case directory:

```bash
/path/to/build/lib/g2d.exe
```

### Python solver API

```python
import numpy as np
from garfoil import run, vary_Re_with_Mach

# Load grid (Plot3D-style format: jtot ktot, then x and y coordinates)
with open("airfoil.grid") as f:
    jtot, ktot = [int(x) for x in f.readline().split()]
xy = np.loadtxt("airfoil.grid", skiprows=1).reshape(2, ktot, jtot)
xy = np.ascontiguousarray(xy.transpose((1, 2, 0)))

machs = np.array([0.3], dtype="d")
reys  = np.array([2e6], dtype="d")
aoas  = np.linspace(-10, 20, 31, dtype="d")

vary_Re_with_Mach(True)
run(machs, reys, aoas, xy, "sa", "my_airfoil", 5, 0)
#                                  eqns  name  order gpu_index
```

### Mesh generation (Python)

```python
from ogen import MeshGen
import numpy as np

# airfoil_xy: (N, 2) array of airfoil surface coordinates
inputs = {
    "ds0":     0.0001,  # wall-normal spacing at surface
    "ktot":    256,      # number of points in normal direction
    "stretch": 1.15,     # growth ratio (or use "far" for auto)
    "far":     50.0,     # far-field distance in chords
    "omega":   1.5,      # SOR relaxation parameter
}

mg = MeshGen(airfoil_xy, inputs)
mg.poisson(100, 1.5)                # smooth grid: iterations, omega
grid = mg.get_mesh()                 # returns (ktot, jtot, 2) array
mg.write_to_file("output.grid")
```

### Airfoil generation utilities

The `meshgen/python/` directory includes tools for creating and manipulating airfoil geometries:

- `naca.py` -- NACA 4- and 5-digit airfoil coordinate generation
- `hickshenne.py` -- Hicks-Henne bump functions for shape perturbation
- `airfoil_utils.py` -- airfoil I/O and manipulation
- `close.py` -- trailing edge closure methods
- `spandist.py` -- point distribution functions

## Solver output

Results are written to a directory named after the airfoil. For each flow condition:

| File                            | Contents                        |
|---------------------------------|---------------------------------|
| `<name>_r<Re>_a<AoA>_m<M>.res`     | Residual history                |
| `<name>_r<Re>_a<AoA>_m<M>.forces`  | Cl, Cd, Cm convergence history  |
| `<name>_r<Re>_a<AoA>_m<M>.cpcf`    | Cp and Cf surface distributions |
| `<name>_r<Re>_a<AoA>_m<M>_sol.dat` | Full solution field             |

## Examples

See `rundirs/` for example cases:

- `7a_example/` -- multi-Mach RANS sweep of a rotor airfoil
- `opt_design_1/` -- Python-driven optimization workflow

## License

MIT
