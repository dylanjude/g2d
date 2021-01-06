#include "g2d.h"
#include "gpu.h"

__global__ void init_flow(int jtot,int ktot,int nvar, double* q, double* machs, double* aoas){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int l  = blockIdx.z;

  double aoa  = aoas[l];
  double M    = machs[l];
  double rinf = 1.0;
  double pinf = 1.0/GAMMA;
  double uinf = M*cos(aoa*PI/180);
  double vinf = M*sin(aoa*PI/180);

  if(j<jtot and k<ktot){
    q += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
    q[0] = rinf;
    q[1] = uinf*rinf;
    q[2] = vinf*rinf;
    q[3] = pinf/(GAMMA-1) + 0.5*rinf*(uinf*uinf+vinf*vinf);
    q[4] = (k > ktot-3)? 0.1 : 3.0; // maybe gradually ramp?
  }

}

void G2D::init(){

  int j,k,idx1, idx2, idx3;

  int qcount = nl*jtot*ktot*nvar;

  this->x[CPU] = new double2[jtot*ktot];

  HANDLE_ERROR( cudaMalloc((void**)&this->x[GPU], jtot*ktot*sizeof(double2)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->q[GPU], qcount*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->qp,     qcount*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->qsafe,  qcount*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->s,      qcount*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->wrk,  4*qcount*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->dt,     jtot*ktot*nl*sizeof(double)) );
  if(this->eqns != EULER){
    HANDLE_ERROR( cudaMalloc((void**)&this->mulam,  jtot*ktot*nl*sizeof(double)) );
    HANDLE_ERROR( cudaMalloc((void**)&this->muturb, jtot*ktot*nl*sizeof(double)) );
    HANDLE_ERROR( cudaMemset(this->muturb, 0, jtot*ktot*nl*sizeof(double)) );
  }

  for(k=nghost; k<ktot-nghost+1; k++){
    for(j=nghost; j<jtot-nghost+1; j++){
      idx1 = j + k*jtot;
      idx2 = (j-nghost) + (k-nghost)*(jtot-2*nghost+1);
      x[CPU][idx1].x = x0[idx2].x;
      x[CPU][idx1].y = x0[idx2].y;
    }
  }

  // Periodic Jmin, Jmax ghost coords
  for(k=0; k<ktot; k++){
    for(j=0; j<nghost; j++){
      idx1 = j + k*jtot;
      idx2 = (jtot-2*nghost+j) + k*jtot;
      x[CPU][idx1].x = x[CPU][idx2].x;
      x[CPU][idx1].y = x[CPU][idx2].y;
    }
    for(j=jtot-nghost; j<jtot; j++){
      idx1 = j + k*jtot;
      idx2 = (j-jtot+2*nghost) + k*jtot;
      x[CPU][idx1].x = x[CPU][idx2].x;
      x[CPU][idx1].y = x[CPU][idx2].y;
    }
  }
  
  // Extrapolate Kmin, Kmax ghost coords
  for(j=0; j<jtot; j++){
    for(k=0; k<nghost; k++){
      idx1 = j + k*jtot;
      idx2 = j + (nghost)*jtot;
      idx3 = j + (2*nghost-k)*jtot;
      x[CPU][idx1].x = 2*x[CPU][idx2].x - x[CPU][idx3].x;
      x[CPU][idx1].y = 2*x[CPU][idx2].y - x[CPU][idx3].y;
    } 
    for(k=ktot-nghost+1; k<ktot; k++){
      idx1 = j + k*jtot;
      idx2 = j + (ktot-nghost)*jtot;
      idx3 = j + (2*(ktot-nghost)-k)*jtot;
      x[CPU][idx1].x = 2*x[CPU][idx2].x - x[CPU][idx3].x;
      x[CPU][idx1].y = 2*x[CPU][idx2].y - x[CPU][idx3].y;
    }
  }

  HANDLE_ERROR( cudaMemcpy(x[GPU], x[CPU], jtot*ktot*sizeof(double2), cudaMemcpyHostToDevice) );

  dim3 thr(32,16,1);
  dim3 blk;
  blk.x = (jtot-1)/thr.x+1;
  blk.y = (ktot-1)/thr.y+1;
  blk.z = nl;

  init_flow<<<blk,thr>>>(jtot,ktot,nvar,q[GPU],machs[GPU],aoas[GPU]);

  HANDLE_ERROR( cudaMemcpy(qsafe, q[GPU], qcount*sizeof(double), cudaMemcpyHostToDevice) );  

  this->metrics();

  // Set viscosities and apply an initial boundary condition
  if(this->eqns != EULER)     this->set_mulam(q[GPU]);
  if(this->eqns == TURBULENT) this->set_muturb(q[GPU]);
  this->apply_bc(istep, this->q[GPU]);

  // Set viscosities and apply an initial boundary condition
  if(this->eqns != EULER)     this->set_mulam(q[GPU]);
  if(this->eqns == TURBULENT) this->set_muturb(q[GPU]);
  this->apply_bc(istep, this->q[GPU]);

}


