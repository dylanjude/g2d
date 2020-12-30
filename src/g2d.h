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

#define AVG_HIST 20

#define CFL_TIMEACC 1.0
#define DX_TIMEACC  0.01

#define DT_GLOBAL( m ) (DX_TIMEACC*CFL_TIMEACC/(1+m))
// #define DT_GLOBAL( m ) (0.001)

#include "gpu.h"
#include "helper_math.h"
#include "timer.h"

class G2D {

  int nM,nRey,nAoa,nl;
  double* machs[2];
  double* reys[2];
  double* aoas[2];
  double* lreys[2];

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

  FILE** resfile;

  std::string* res_fname;
  std::string* forces_fname;
  std::string* cpcf_fname;
  std::string* sol_fname;

  double *res0, *res;
  double *fhist; // force history

  bool timeac;
  int eqns;
  int istep, iforce;

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
  void gmres(double* s, int isub);
  void dadi(double* s);
  void check_convergence();
  void check_forces();
  void write_cpcf();
  void compute_residual(double* s, int isub);

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

  void take_steps(int ns, int nsub, double cfl);

 public:
  G2D(int nM,int nR,int nAoA,int jtot,int ktot, int order, double* machs,double* reys,double* aoas,
      double* xy,int eqns,std::string aname);
  ~G2D();
  void init();
  void go();
  void write_sols();

};
