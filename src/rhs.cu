#include "g2d.h"

#define DBGJ 191
#define DBGK 50

// add bdf terms and divide by volume
__global__ void bdf(int jtot,int ktot,int nvar,int nghost, double* q, double* qp, double* s, double* dt){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int v  = threadIdx.z;

  q   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  qp  += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  s   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  dt  += j      + k*jtot      + blockIdx.z*jtot*ktot;

  if(j>jtot-1 or k>ktot-1) return;

  if(j<nghost or k<nghost or j>jtot-nghost-1 or k>ktot-nghost-1){
    s[v] = 1e-16;
  } else {
    s[v] = s[v] - (q[v]-qp[v])/dt[0];
  }
  // if(j+nghost < jtot and k+nghost < ktot){

  // }

}

void G2D::compute_rhs(double* qtest, double* stest){

  int nl     = nM*nRey*nAoa;
  int qcount = nl*jtot*ktot*nvar;

  dim3 vthr(32,4,nvar); // turb var already divided by volume
  dim3 vblk;
  vblk.x = (jtot-1)/vthr.x+1;
  vblk.y = (ktot-1)/vthr.y+1;
  vblk.z = nl;

  HANDLE_ERROR( cudaMemset(stest, 0, qcount*sizeof(double)) );

  // Set viscosity based on provided Q
  if(this->eqns != EULER)     this->set_mulam(qtest);
  if(this->eqns == TURBULENT) this->set_muturb(qtest);

  this->apply_bc(istep);

  // Inviscid Fluxes
  this->inviscid_flux(qtest,stest);

  // debug_print(87,3,0,qtest,5);

  // Viscous Fluxes
  if(this->eqns != EULER){
    this->viscous_flux(qtest,stest);
  }

  // Turbulence Model
  if(this->eqns == TURBULENT){
    this->sa_rhs(qtest, stest);
  }

  bdf<<<vblk,vthr>>>(jtot,ktot,nvar,nghost,qtest,qp,stest,dt);

}
