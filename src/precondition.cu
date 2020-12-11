#include "g2d.h"

__global__ void times_dt(int jtot,int ktot,int nvar,int nghost,double* s, double* dt){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;
  int v  = threadIdx.z;

  s   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  dt  += j      + k*jtot      + blockIdx.z*jtot*ktot;

  if(j+nghost < jtot and k+nghost < ktot){
    // for(int v=0; v<nvar; v++){
    s[v] *= dt[0];
    // }
  }
}

void G2D::precondition(double* sin, double* sout){

  int nl     = nM*nRey*nAoa;
  int qcount = nl*jtot*ktot*nvar;
  int count4 = nl*jtot*ktot*4;

  if(sin != sout){
    HANDLE_ERROR( cudaMemcpy(sout, sin, qcount*sizeof(double), cudaMemcpyDeviceToDevice) );
  }

  dim3 vthr(32,4,nvar);
  dim3 vblk;
  vblk.x = (jtot-1-nghost*2)/vthr.x+1;
  vblk.y = (ktot-1-nghost*2)/vthr.y+1;
  vblk.z = nl;

  times_dt<<<vblk,vthr>>>(jtot,ktot,nvar,nghost,sout,dt);

  // Precondition mean-flow equations
  this->dadi(sout);

}
