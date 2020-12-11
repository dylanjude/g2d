#pragma once

#include <string>

#define CPU 0
#define GPU 1

#define GAMMA        1.4
#define PRANDTL      0.72
#define TURB_PRANDTL 0.9

#define PI 3.1415926535897931

#define JMIN_FACE 0
#define JMAX_FACE 1
#define KMIN_FACE 2
#define KMAX_FACE 3

#include "gpu.h"
#include "helper_math.h"

class G2D {

  int nM,nRey,nAoa;
  double* machs[2];
  double* reys[2];
  double* aoas[2];

  int order, nghost, nvar;
  int jtot,ktot;

  double2* x0;
  double2* x[2];
  double* q[2];

  double *qp, *dt, *wrk, *s;
  double *mulam;

  double2 *Sj, *Sk;
  double* vol;

  double *xi, *eta;

  void apply_bc(int istep);
  void metrics();
  void inviscid_flux(double* q, double* s);
  void viscous_flux(double* q, double* s);
  void compute_rhs(double* q, double* s);
  void precondition(double* sin, double* sout);
  void check_convergence(double* s);

  void debug_print(int j, int k, int l, double* v, int nvar);

 public:
  G2D(int nM,int nR,int nAoA,int jtot,int ktot, int order, double* machs,double* reys,double* aoas,double* xy);
  ~G2D();
  void init();
  void go();
  void write_sols();

};
