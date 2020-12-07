#include "g2d.h"
#include "gpu.h"

__global__ void compute_metrics(int jtot,int ktot,int nvar,double2* Sj,double2* Sk){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  int jp, kp;
  int jj, kk;

  jj = min(j, jtot-2);
  kk = min(k, ktot-2); // safe second-to-last index

  jp = min(j+1, jtot-1);
  kp = min(k+1, ktot-1); // safe last index

  if(j > jtot-1 or k > ktot-1) return;

  int idx = j + k*jtot;

  int idx1, idx2;

  idx1 = j + kk*jtot;
  idx2 = j + kp*jtot;

  // double aoa  = aoas[ia];
  // double M    = machs[im];
  // double rinf = 1.0;
  // double pinf = 1.0/GAMMA;
  // double uinf = M*cos(aoa*2*PI/180);
  // double vinf = M*sin(aoa*2*PI/180);

  // if(j<jtot and k<ktot){
  //   q += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  //   q[0] = rinf;
  //   q[1] = uinf*rinf;
  //   q[2] = vinf*rinf;
  //   q[3] = pinf/(GAMMA-1) + 0.5*rinf*(uinf*uinf+vinf*vinf);
  //   q[4] = 3.0;
  // }

}

void G2D::metrics(){

  double2 d;

  d.x = 4.0;

  // int nl     = nM*nRey*nAoa;
  // int qcount = nl*jtot*ktot*nvar;

  // dim3 thr(32,16,1);
  // dim3 blk;
  // blk.x = (jtot-1)/thr.x+1;
  // blk.y = (ktot-1)/thr.y+1;
  // blk.z = nl;

  // init_flow<<<blk,thr>>>(jtot,ktot,nvar,nM,nAoa,nRey,q[GPU],machs[GPU],aoas[GPU]);

}
