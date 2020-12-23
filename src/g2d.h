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

#define EULER     0
#define LAMINAR   1
#define TURBULENT 2

#define SASCALE 0.01

#include "gpu.h"
#include "helper_math.h"
#include "timer.h"

class G2D {

  int nM,nRey,nAoa;
  double* machs[2];
  double* reys[2];
  double* aoas[2];

  int order, nghost, nvar;
  int jtot,ktot;

  double2* x0;
  double2* x[2];
  double2* xc;
  double* q[2];

  double *qp, *dt, *wrk, *s;
  double *mulam, *muturb;

  double2 *Sj, *Sk;
  double* vol;

  double *xi, *eta;

  FILE* resfile;

  int eqns;
  int istep;

  int debug_flag;

  int gmres_nkrylov;
  double *gmres_r, *gmres_Av, *gmres_h, *gmres_g, *gmres_v, *gmres_giv, *gmres_scr;

  Timer timer;

  void apply_bc(int istep, double* q);
  void zero_bc(double* s);
  void metrics();
  void inviscid_flux(double* q, double* s);
  void viscous_flux(double* q, double* s);
  void compute_rhs(double* q, double* s);
  void precondition(double* sin, double* sout);
  void gmres(double* s);
  void dadi(double* s);
  void check_convergence(double* s);

  void vdp(double* a, double* b, double* out);
  void mvp(double* a, double* b);
  void axpy(double* a, double* x, double* y, double* out);
  void l2norm(double* vec, double* l2);

  void sa_rhs(double* q, double* s);
  void sa_adi(double* s);

  void set_mulam(double* q);
  void set_muturb(double* q);
  void compute_vorticity(double* q, double* vort);

  void debug_print(int j, int k, int l, double* v, int nvar);

 public:
  G2D(int nM,int nR,int nAoA,int jtot,int ktot, int order, double* machs,double* reys,double* aoas,double* xy,int eqns);
  ~G2D();
  void init();
  void go();
  void write_sols();

};
