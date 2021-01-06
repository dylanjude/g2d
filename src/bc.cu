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
    q[idx+0] =  q[midx+0];   // rho
    q[idx+1] = -q[midx+1];   // rho*(-u)
    q[idx+2] = -q[midx+2];   // rho*(-v)
    q[idx+3] =  q[midx+3];   // e
    if(nvar==5){ 
      q[idx+4] = -q[midx+4]; // -nu_tilde
    }
  }

}

template<int face>
__global__ void bc_inv_wall(int jtot,int ktot,int nvar,int nghost,double* q,double2* Sk){

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

  q[idx+0] = q[midx+0];                          // rho
  q[idx+1] = rhou-2.0*wall_vector.x*v_dot_wall;  // rho-u
  q[idx+2] = rhov-2.0*wall_vector.y*v_dot_wall;  // rho-v
  q[idx+3] = q[midx+3];                          // e

  if(nvar==5){ 
    q[idx+4] = -q[midx+4]; // -nu_tilde
  }

}

template<int face>
__global__ void bc_far(int jtot,int ktot,int nvar,int nghost,double* q,double2* Sk){

  int j,k;

  if(face == KMAX_FACE){
    j  = blockDim.x*blockIdx.x + threadIdx.x;
    k  = ktot-threadIdx.y-1;
  } else {
    printf("BC far not implemented for this face\n");
    return;
  }

  if(j>jtot-1 or k>ktot-1) return;

  int idx   = (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;
  int cidx1 = (j + (ktot-nghost-1)*jtot + blockIdx.z*jtot*ktot)*nvar;
  int cidx2 = (j + (ktot-nghost-2)*jtot + blockIdx.z*jtot*ktot)*nvar;

  double2 out_vector = Sk[j + (ktot-nghost)*jtot]; // kmax face

  // double imag   = 1.0/sqrt(dot(out_vector, out_vector));
  // out_vector.x = out_vector.x * imag;
  // out_vector.y = out_vector.y * imag;

  double u_in    = q[cidx1+1]/q[cidx1];
  double v_in    = q[cidx1+2]/q[cidx1];

  double dotted  = u_in*out_vector.x + v_in*out_vector.y;
  
  if(dotted<0.0){ // inflow, use freestream
    q[idx+4] = 0.1; // nu turb infinity;
  } else {
    q[idx+4] = 2*q[cidx1+4] - q[cidx2+4];
  }

  // everything else should not change

}

template<int dir>
__global__ void bc_zero(int jtot,int ktot,int nvar,int nghost,double* s){

  int j,k;
  int v = threadIdx.z;

  if(dir==0){
    j  = (jtot+threadIdx.y-nghost)%jtot;
    k  = blockDim.x*blockIdx.x + threadIdx.x;
  } else {
    k  = (ktot+threadIdx.y-nghost)%ktot;
    j  = blockDim.x*blockIdx.x + threadIdx.x;
  }

  if(j>jtot-1 or k>ktot-1) return;

  s += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;

  s[v] = 0;
}

template<int face>
__global__ void bc_ramp(int jtot,int ktot,int nvar,int nghost, double* q,double* machs, double* aoas, double ratio){
  int j,k;
  if(face == KMIN_FACE){
    j  = blockDim.x*blockIdx.x + threadIdx.x;
    k  = threadIdx.y;
  } else {
    printf("BC visc wall not implemented for this face\n");
    return;
  }

  int l = blockIdx.z;

  if(j>jtot-1 or k>ktot-1) return;

  q += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;

  double aoa  = aoas[l];
  double M    = machs[l];
  double q0 = 1.0;
  double q1 = M*cos(aoa*PI/180);
  double q2 = M*sin(aoa*PI/180);
  double p  = 1.0/GAMMA;
  double q3 = p/(GAMMA-1) + 0.5*(q1*q1+q2*q2); // assume q0 = 1.0
  double q4 = (k > ktot-3)? 0.1 : 3.0; // maybe gradually ramp?

  q[0] = (1-ratio)*q0 + (ratio)*q[0];
  q[1] = (1-ratio)*q1 + (ratio)*q[1];
  q[2] = (1-ratio)*q2 + (ratio)*q[2];
  q[3] = (1-ratio)*q3 + (ratio)*q[3];

  if(nvar==5){ 
    q[4] = (1-ratio)*q4 + (ratio)*q[4];
  }

}

void G2D::apply_bc(int istep, double* qtest){

  dim3 thr(32,nghost,nvar);
  dim3 blkj(1,1,nl), blkk(1,1,nl);

  blkj.x = (ktot-1)/thr.x+1; // j-face bc (k-varying)
  blkk.x = (jtot-1)/thr.x+1; // k-face bc (j-varying)

  bc_periodic<JMIN_FACE><<<blkj,thr>>>(jtot,ktot,nvar,nghost,qtest);
  bc_periodic<JMAX_FACE><<<blkj,thr>>>(jtot,ktot,nvar,nghost,qtest);

  thr.z = 1;

  if(this->eqns == EULER){
    bc_inv_wall<KMIN_FACE><<<blkk,thr>>>(jtot,ktot,nvar,nghost,qtest,Sk);
  } else {
    bc_visc_wall<KMIN_FACE><<<blkk,thr>>>(jtot,ktot,nvar,nghost,qtest);
  }

  // double ratio = 1.0;
  // if(istep < 30){
  //   ratio = ( istep*1.0 )/( 30.0 );                                                                                            
  //   ratio = (10.0 - 15.0*ratio + 6.0*ratio*ratio)*ratio*ratio*ratio; 
  //   bc_ramp<KMIN_FACE><<<blkk,thr>>>(jtot,ktot,nvar,nghost,qtest,machs[GPU],aoas[GPU],ratio);
  // }

  bc_far<KMAX_FACE><<<blkk,thr>>>(jtot,ktot,nvar,nghost,qtest,Sk);

}

void G2D::zero_bc(double* s){

  dim3 thr(16,nghost*2,nvar);
  dim3 blkj(1,1,nl), blkk(1,1,nl);

  blkj.x = (ktot-1)/thr.x+1; // j-face bc (k-varying)
  blkk.x = (jtot-1)/thr.x+1; // k-face bc (j-varying)

  bc_zero<0><<<blkj,thr>>>(jtot,ktot,nvar,nghost,s);
  bc_zero<1><<<blkk,thr>>>(jtot,ktot,nvar,nghost,s);

}
