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

  int idx  = (j + k*jtot)*nvar + blockIdx.z*jtot*ktot*nvar + v;
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
__global__ void bc_visc_wall(int jtot,int ktot,int nvar,int nghost,double* q,double ratio){

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
    if(ratio > 0.99){
      q[idx+0] =  q[midx+0];  // rho
      q[idx+1] = -q[midx+1];  // rho*(-u)
      q[idx+2] = -q[midx+2];  // rho*(-v)
      q[idx+3] =  q[midx+3];  // e
    } else { // ramp
      q[idx+0] = (1-ratio)*q[idx+0] + (ratio)*q[midx+0];  // rho
      q[idx+1] = (1-ratio)*q[idx+1] - (ratio)*q[midx+1];  // rho*(-u)
      q[idx+2] = (1-ratio)*q[idx+2] - (ratio)*q[midx+2];  // rho*(-v)
      q[idx+3] = (1-ratio)*q[idx+3] + (ratio)*q[midx+3];  // e
    }
    // dont ramp turbulence
    if(nvar==5){ 
      q[idx+4] = -q[midx+4]; // -nu_tilde
    }
  }

}

template<int face>
__global__ void bc_inv_wall(int jtot,int ktot,int nvar,int nghost,double* q,double2* Sk, double ratio){

  int j,k;

  if(face == KMIN_FACE){
    j  = blockDim.x*blockIdx.x + threadIdx.x;
    k  = threadIdx.y;
  } else {
    printf("BC visc wall not implemented for this face\n");
    return;
  }

  if(j>jtot-1 or k>ktot-1) return;

  int idx  = (j + k*jtot)*nvar + blockIdx.z*jtot*ktot*nvar;
  int midx = (j + (2*(nghost)-k-1)*jtot)*nvar + blockIdx.z*jtot*ktot*nvar;

  double2 wall_vector = Sk[j + nghost*jtot]; // for kmin face

  double imag   = 1.0/sqrt(dot(wall_vector, wall_vector));
  wall_vector.x = wall_vector.x * imag;
  wall_vector.y = wall_vector.y * imag;

  double rhou    = q[midx+1];
  double rhov    = q[midx+2];

  double v_dot_wall = wall_vector.x*rhou + wall_vector.y*rhov;

  if(ratio > 0.99){
    q[idx+0] = q[midx+0];                          // rho
    q[idx+1] = rhou-2.0*wall_vector.x*v_dot_wall;  // rho-u
    q[idx+2] = rhov-2.0*wall_vector.y*v_dot_wall;  // rho-v
    q[idx+3] = q[midx+3];                          // e
  } else {
    q[idx+0] =  (1-ratio)*q[idx+0] + (ratio)*q[midx+0];                          // rho
    q[idx+1] =  (1-ratio)*q[idx+1] + (ratio)*rhou-2.0*wall_vector.x*v_dot_wall;  // rho-u
    q[idx+2] =  (1-ratio)*q[idx+2] + (ratio)*rhov-2.0*wall_vector.y*v_dot_wall;  // rho-v
    q[idx+3] =  (1-ratio)*q[idx+3] + (ratio)*q[midx+3];                          // e
  }
  // dont ramp turbulence
  if(nvar==5){ 
    q[idx+4] = -q[midx+4]; // -nu_tilde
  }

}



void G2D::apply_bc(int istep){

  int nl     = nM*nRey*nAoa;

  dim3 thr(32,nghost,nvar);
  dim3 blkj(1,1,nl), blkk(1,1,nl);

  blkj.x = (ktot-1)/thr.x+1; // j-face bc (k-varying)
  blkk.x = (jtot-1)/thr.x+1; // k-face bc (j-varying)

  bc_periodic<JMIN_FACE><<<blkj,thr>>>(jtot,ktot,nvar,nghost,q[GPU]);
  bc_periodic<JMAX_FACE><<<blkj,thr>>>(jtot,ktot,nvar,nghost,q[GPU]);

  thr.z = 1;

  double ratio = 1.0;
  if(istep < 30){
    ratio = ( istep*1.0 )/( 30.0 );                                                                                            
    ratio = (10.0 - 15.0*ratio + 6.0*ratio*ratio)*ratio*ratio*ratio; 
  }

  // bc_visc_wall<KMIN_FACE><<<blkk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],ratio);
  bc_inv_wall<KMIN_FACE><<<blkk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],Sk,ratio);

  // bc_far<KMAX_FACE><<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU]);

}
