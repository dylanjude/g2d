#include "g2d.h"
#include "gpu.h"

template<int face>
__global__ void bc_periodic(int jtot,int ktot,int nvar,int nghost,double* q){

  int j,k,v;

  if(face == JMIN_FACE){
    k  = blockDim.x*blockIdx.x + threadIdx.x;
    j  = threadIdx.y;
  } else if(face == JMAX_FACE){
    k  = blockDim.x*blockIdx.x + threadIdx.x;
    j  = jtot-threadIdx.y-1;
  } else {
    printf("BC periodic not implemented for this face\n");
    return;
  }
  
  v = threadIdx.z;

  int idx  = (j + k*jtot)*nvar + blockIdx.z*jtot*ktot*nvar;
  int pidx;
  
  if(face == JMIN_FACE){
    pidx = ( (jtot-2*nghost+j) + k*jtot )*nvar + blockIdx.z*jtot*ktot*nvar + v;
  } else if(face == JMAX_FACE){
    pidx = ( (j-jtot+2*nghost) + k*jtot )*nvar + blockIdx.z*jtot*ktot*nvar + v;
  }

  if(j<jtot and k<ktot){
    q[idx] = q[pidx];
  }
			    
}

template<int face>
__global__ void bc_visc_wall(int jtot,int ktot,int nvar,int nghost,double* q){

  int j,k;

  if(face == KMIN_FACE){
    j  = blockDim.x*blockIdx.x + threadIdx.x;
    k  = threadIdx.y;
  } else {
    printf("BC visc wall not implemented for this face\n");
    return;
  }

  int idx  = (j + k*jtot)*nvar + blockIdx.z*jtot*ktot*nvar;
  int midx = (j + (2*(nghost)-k-1)*jtot)*nvar + blockIdx.z*jtot*ktot*nvar;

  if(j<jtot and k<ktot){
    q[idx+0] =  q[midx+0];  // rho
    q[idx+1] = -q[midx+1];  // rho*(-u)
    q[idx+2] = -q[midx+2];  // rho*(-v)
    q[idx+3] =  q[midx+3];  // e
    q[idx+4] = -q[midx+4];  // -nu_tilde
  }

}


void G2D::apply_bc(){

  int nl     = nM*nRey*nAoa;
  int qcount = nl*jtot*ktot*nvar;

  dim3 thr(32,nghost,nvar);
  dim3 blkj(1,1,nl), blkk(1,1,nl);

  blkj.x = (ktot-1)/thr.x+1; // j-face bc (k-varying)
  blkk.x = (jtot-1)/thr.x+1; // k-face bc (j-varying)

  bc_periodic<JMIN_FACE><<<blkj,thr>>>(jtot,ktot,nvar,nghost,q[GPU]);
  bc_periodic<JMAX_FACE><<<blkj,thr>>>(jtot,ktot,nvar,nghost,q[GPU]);

  thr.z = 1;

  bc_visc_wall<KMIN_FACE><<<blkk,thr>>>(jtot,ktot,nvar,nghost,q[GPU]);

  // bc_far<KMAX_FACE><<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU]);

}
