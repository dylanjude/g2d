#include "g2d.h"
#include "gpu.h"

void G2D::metrics(){

  int j,k,idx;
  int jp, kp;

  double2 vec;
  
  double2 x1, x2, x3, x4;

  // We'll just do the metrics on the CPU and copy them to the GPU once.
  if(Sj)  HANDLE_ERROR( cudaFree(Sj) );
  if(Sk)  HANDLE_ERROR( cudaFree(Sk) );
  if(vol) HANDLE_ERROR( cudaFree(vol) );

  Sj = new double2[jtot*ktot];
  Sk = new double2[jtot*ktot];
  vol = new double[jtot*ktot];

  for(k=0;k<ktot-1;k++){
    kp=k+1;
    for(j=0;j<jtot-1;j++){
      jp=j+1;

      x1 = x[CPU][j+k*jtot];
      x2 = x[CPU][jp+k*jtot];
      x3 = x[CPU][jp+kp*jtot];
      x4 = x[CPU][j+kp*jtot];

      // J-direction-face (k-varying)
      vec = x4-x1;
      Sj[j+k*jtot].x = -vec.y;
      Sj[j+k*jtot].y =  vec.x;
      // K-direction-face (j-varying)
      vec = x2-x1;
      Sk[j+k*jtot].x = -vec.y;
      Sk[j+k*jtot].y =  vec.x;
      
      vol[j+k*jtot] = 0.5*((x1.x-x3.x)*(x2.y-x4.y)+(x4.x-x2.x)*(x1.y-x3.y));

    }
  }

  k=ktot-1;
  for(j=0;j<jtot-1;j++){
    jp=j+1;
    x1 = x[CPU][j+k*jtot];
    x2 = x[CPU][jp+k*jtot];
    // K-direction-face (j-varying)
    vec = x2-x1;
    Sk[j+k*jtot].x = -vec.y;
    Sk[j+k*jtot].y =  vec.x;
    vol[j+k*jtot]  = vol[j+(k-1)*jtot];
  }
  j=jtot-1;
  for(k=0;k<ktot-1;k++){
    kp = k+1;
    x1 = x[CPU][j+k*jtot];
    x4 = x[CPU][j+kp*jtot];
    // J-direction-face (k-varying)
    vec = x4-x1;
    Sj[j+k*jtot].x = -vec.y;
    Sj[j+k*jtot].y =  vec.x;
  }

  j=jtot-1;
  k=ktot-1;
  vol[j+k*jtot]  = vol[(j-1)+(k-1)*jtot];

  double2 *tmp2;
  double  *tmp;

  // Copy Sj to GPU
  tmp2 = Sj;
  HANDLE_ERROR( cudaMalloc((void**)&Sj, jtot*ktot*sizeof(double2)) );
  HANDLE_ERROR( cudaMemcpy(Sj, tmp2, jtot*ktot*sizeof(double2), cudaMemcpyHostToDevice) );
  delete[] tmp2;

  // Copy Sk to GPU
  tmp2 = Sk;
  HANDLE_ERROR( cudaMalloc((void**)&Sk, jtot*ktot*sizeof(double2)) );
  HANDLE_ERROR( cudaMemcpy(Sk, tmp2, jtot*ktot*sizeof(double2), cudaMemcpyHostToDevice) );
  delete[] tmp2;
  
  // Copy vol to GPU
  tmp = vol;
  HANDLE_ERROR( cudaMalloc((void**)&vol, jtot*ktot*sizeof(double)) );
  HANDLE_ERROR( cudaMemcpy(vol, tmp, jtot*ktot*sizeof(double), cudaMemcpyHostToDevice) );
  delete[] tmp;

}
