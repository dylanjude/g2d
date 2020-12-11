#include "g2d.h"

#define DBGJ 191
#define DBGK 50

// add bdf terms and divide by volume
__global__ void bdf(int jtot,int ktot,int nvar,int nghost, double* q, double* qp, double* s, double* dt, double *vol){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;
  int v  = threadIdx.z;

  q   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  qp  += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  s   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  dt  += j      + k*jtot      + blockIdx.z*jtot*ktot;
  vol += j      + k*jtot;

  if(j+nghost < jtot and k+nghost < ktot){
    s[v] = s[v]/vol[0] - (q[v]-qp[v])/dt[0];
  }

}

void G2D::compute_rhs(double* qtest, double* stest){

  int nl     = nM*nRey*nAoa;
  int qcount = nl*jtot*ktot*nvar;

  dim3 vthr(32,4,nvar);
  dim3 vblk;
  vblk.x = (jtot-1-2*nghost)/vthr.x+1;
  vblk.y = (ktot-1-2*nghost)/vthr.y+1;
  vblk.z = nl;

  HANDLE_ERROR( cudaMemset(stest, 0, qcount*sizeof(double)) );

  inviscid_flux(qtest,stest);

  if(this->eqns != EULER){
    viscous_flux(qtest,stest);
  }

  bdf<<<vblk,vthr>>>(jtot,ktot,nvar,nghost,qtest,qp,stest,dt,vol);

  // sa_rhs(qtest,stest);

}
