#include "g2d.h"

__global__ void square_array(int jtot, int ktot, int nvar, int nghost, double* s, double* wrk){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int v  = threadIdx.z;

  int lin_idx = j + k*(jtot-nghost*2) + blockIdx.z*(jtot-nghost*2)*(ktot-nghost*2);
  lin_idx    *= nvar;

  j += nghost;
  k += nghost;

  if(j+nghost > jtot-1 or k+nghost > ktot-1) return;

  s     += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  wrk   += lin_idx;

  wrk[v] = s[v]*s[v];

}

__global__ void sum1(double* a, int n, double* b){
  
  extern __shared__ double ish1[];
  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(i<n){
     ish1[tid] = a[i];
  } else {
     ish1[tid] = 0;
  }

  __syncthreads();

  // initial increment is half the block dimension,
  // then 1/4, 1/8 etc. ">>" is a shift operator for
  // fast binary operations. >>1 shifts by 1 bit is
  // equivalent to a divide by two.
  for(int s=blockDim.x/2; s>0; s>>=1){
    if(tid < s){
      ish1[tid] += ish1[tid+s];
    }
    __syncthreads();
  }

  if(tid == 0){ 
    b[blockIdx.x] = ish1[0];
  } 

}


void G2D::check_convergence(int istep, double* s){

  int nl        = nM*nRey*nAoa;
  int qcount    = nl*jtot*ktot*nvar;
  int i, leftover, smem;

  dim3 vthr(32,4,nvar);
  dim3 vblk;
  vblk.x = (jtot-1-nghost*2)/vthr.x+1;
  vblk.y = (ktot-1-nghost*2)/vthr.y+1;
  vblk.z = nl;

  int idof = (jtot-nghost*2)*(ktot-nghost*2)*nl;

  int c=0;
  double* scratch1 = &wrk[c]; c+= idof;
  double* scratch2 = &wrk[c]; c+= idof;
 
  square_array<<<vblk,vthr>>>(jtot,ktot,nvar,nghost,s,scratch1);

  dim3 threads(1,1,1), blocks(1,1,1);

  int n     = idof;
  int power = min(9, (int)ceil(log2(n*1.0)));

  double normsq;

  threads.x = pow(2,power);
  blocks.y  = 1;
  leftover = n;
  i = 0;
  while(leftover > 1){

    blocks.x = (leftover - 1)/ threads.x + 1;
    smem = threads.x*sizeof(double);
    
    if(i%2 == 0){
      sum1<<<blocks,threads,smem>>>(scratch1, leftover, scratch2);
    } else {
      sum1<<<blocks,threads,smem>>>(scratch2, leftover, scratch1);
    }
    i++;
    leftover = blocks.x;
  }

  if(i%2 == 1){
    HANDLE_ERROR(cudaMemcpy(&normsq,scratch2,sizeof(double),cudaMemcpyDeviceToHost));
  } else {
    HANDLE_ERROR(cudaMemcpy(&normsq,scratch1,sizeof(double),cudaMemcpyDeviceToHost));
  }

  printf("[All Norm] %6d %16.8e\n", istep, sqrt(normsq/idof));

}
