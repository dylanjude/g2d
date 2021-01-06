#include "g2d.h"

#define DBGJ 191
#define DBGK 50

// add bdf terms and divide by volume
__global__ void bdf(int jtot,int ktot,int nvar,int nghost, double* q, double* qp, double* s, 
		    double* dt, double* vol, double* machs, unsigned char* flags){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;
  int v  = threadIdx.z;

  // exit early if this 2D plane is not timeaccurate
  if(!(flags[blockIdx.z] & F_TIMEACC)) return;

  q   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  qp  += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  s   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  dt  += j      + k*jtot      + blockIdx.z*jtot*ktot;

  if(j+nghost>jtot-1 or k+nghost>ktot-1) return;

  double scale = (v==4)? SASCALE : 1.0;

  double dt_global = DT_GLOBAL(machs[blockIdx.z]);

  s[v] = s[v] - scale*(q[v]-qp[v])/dt_global;

}

void G2D::compute_rhs(double* qtest, double* stest){

  int qcount = nl*jtot*ktot*nvar;

  dim3 vthr(32,4,nvar); // turb var already divided by volume
  dim3 vblk;
  vblk.x = (jtot-2*nghost-1)/vthr.x+1;
  vblk.y = (ktot-2*nghost-1)/vthr.y+1;
  vblk.z = nl;

  HANDLE_ERROR( cudaMemset(stest, 0, qcount*sizeof(double)) );

  this->apply_bc(istep, qtest);

  // Set viscosity based on provided Q
  if(this->eqns != EULER)     this->set_mulam(qtest);
  if(this->eqns == TURBULENT) this->set_muturb(qtest);

  // Inviscid Fluxes
  this->inviscid_flux(qtest,stest);

  // Viscous Fluxes
  if(this->eqns != EULER){
    this->viscous_flux(qtest,stest);
  }

  // Turbulence Model
  if(this->eqns == TURBULENT){
    this->sa_rhs(qtest, stest);
  }

  bdf<<<vblk,vthr>>>(jtot,ktot,nvar,nghost,qtest,qp,stest,dt,vol,machs[GPU],flags[GPU]);

  this->zero_bc(stest);

}
