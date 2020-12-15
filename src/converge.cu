#include "g2d.h"

#define GROUP_MEANFLOW

__global__ void reorder_and_square(int jtot, int ktot, int nvar, int nghost, double* s, double* wrk){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int v  = threadIdx.z;

  int lin_idx;

#ifdef GROUP_MEANFLOW
  lin_idx = (j + 
             k*(jtot-nghost*2) + 
	     v*(jtot-nghost*2)*(ktot-nghost*2) +
	     blockIdx.z*2*(jtot-nghost*2)*(ktot-nghost*2));
#else
  lin_idx = (j + 
	     k*(jtot-nghost*2) + 
	     v*(jtot-nghost*2)*(ktot-nghost*2) +
	     blockIdx.z*nvar*(jtot-nghost*2)*(ktot-nghost*2));
#endif

  j += nghost;
  k += nghost;

  if(j+nghost > jtot-1 or k+nghost > ktot-1) return;

  s += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;

#ifdef GROUP_MEANFLOW
  double sum=0;
  if(v==0){
    for(int vv=0; vv<nvar-1; vv++){
      sum += s[vv]*s[vv];
    }
  } else {
    sum = s[nvar-1]*s[nvar-1];
  }
  wrk[lin_idx] = sum;
#else
  wrk[lin_idx] = s[v]*s[v];  
#endif

}

__global__ void sum1(double* a, int n, double* b){
  
  extern __shared__ double ish1[];
  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int v = blockIdx.y;
  int l = blockIdx.z;

  // initially there are n values between each variable on each grid
  if(i<n){
    ish1[tid] = a[i + v*n + l*n*gridDim.y]; // gridDim.y is nvar
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

  // the next "n" will be the number of blocks we have in the x-direction, gridDim.x
  if(tid == 0){ 
    b[blockIdx.x + v*gridDim.x + l*gridDim.x*gridDim.y] = ish1[0];
  } 

}

void G2D::check_convergence(int istep, double* s){

  int nl        = nM*nRey*nAoa;
  // int qcount    = nl*jtot*ktot*nvar;
  int i, leftover, smem;
  int j, k, l, v;

#ifdef GROUP_MEANFLOW
  int nv = 2;
#else
  int nv = nvar;
#endif

  dim3 vthr(32,4,nv);
  dim3 vblk;
  vblk.x = (jtot-1-nghost*2)/vthr.x+1;
  vblk.y = (ktot-1-nghost*2)/vthr.y+1;
  vblk.z = nl;

  int pts = (jtot-nghost*2)*(ktot-nghost*2);

  int c=0;
  double* scratch1 = &wrk[c]; c+= pts*nv*nl;
  double* scratch2 = &wrk[c]; c+= pts*nv*nl;
 
  // --------------------------------------------------------------
  // double* scpu  = new double[jtot*ktot*nvar*nl];
  // HANDLE_ERROR( cudaMemcpy(scpu, s, jtot*ktot*nvar*nl*sizeof(double), cudaMemcpyDeviceToHost) );
  // double l2cpu, ss;
  // int ii;
  // int vcheck=1;
  // int dcnt=1;
  // for(l=0; l<nl; l++){
  //   printf("[cpu] %3d ", l);
  //   for(v=0; v<nvar; v++){
  //     l2cpu=0;
  //     for(k=nghost; k<ktot-nghost; k++){
  // 	for(j=nghost; j<jtot-nghost; j++){
  // 	  ss = scpu[(j + k*jtot + l*jtot*ktot)*nvar+v];
  // 	  // if(ii++<dcnt) printf("%16.8e ", ss*ss);
  // 	  l2cpu += ss*ss;
  // 	}
  //     }
  //     printf("%16.8e ",l2cpu);
  //   }
  //   printf("\n");
  // }
  // delete[] scpu;
  // --------------------------------------------------------------------

  reorder_and_square<<<vblk,vthr>>>(jtot,ktot,nvar,nghost,s,scratch1);

  // HANDLE_ERROR( cudaMemcpy(scpu, scratch1, pts*nv*nl*sizeof(double), cudaMemcpyDeviceToHost) );
  // for(l=0;l<nl;l++){
  //   printf("[cpu]_%3d ", l);
  //   for(v=0; v<nv; v++){
  //     l2cpu=0;
  //     for(i=0; i<pts; i++){
  // 	  ss     = scpu[i + v*pts + l*nv*pts];
  // 	  l2cpu += ss;
  //     }
  //     printf("%16.8e ",l2cpu);
  //   }
  //   printf("\n");
  // }

  dim3 threads(1,1,1), blocks(1,nv,nl);

  double* l2var = new double[nv*nl];

  int n     = pts;
  int power = min(9, (int)ceil(log2(n*1.0)));

  threads.x = pow(2,power);
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
    HANDLE_ERROR(cudaMemcpy(l2var,scratch2,nv*nl*sizeof(double),cudaMemcpyDeviceToHost));
  } else {
    HANDLE_ERROR(cudaMemcpy(l2var,scratch1,nv*nl*sizeof(double),cudaMemcpyDeviceToHost));
  }

  // scratch1 += pts;

  // printf("[All Norm] %6d %16.8e\n", istep, sqrt(normsq/pts));

  // printf("%6d ",istep);

  for(l=0; l<nl; l++){
    printf("%6d %3d ", istep, l);
    for(v=0; v<nv; v++){
      printf("%16.8e ", sqrt(l2var[v + l*nv]));
    }
    // printf("# <-- caution squared! \n");
    printf(" # [gpu]\n");
  }

  // delete[] scpu;
  // delete[] scpu2;

  delete[] l2var;

}
