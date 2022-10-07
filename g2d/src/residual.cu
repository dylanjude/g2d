#include "g2d.h"
#include <cstdio>
#include <cstdlib>

// #define PRINT_STDIO
#define GROUP_MEANFLOW

#define BIGRES 9999999

template<int group>
__global__ void reorder_and_square(int jtot, int ktot, int nvar, int nghost, double* s, double* wrk){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int v  = threadIdx.z;

  int lin_idx;

  if(group){
    lin_idx = (j + 
	       k*(jtot-nghost*2) + 
	       v*(jtot-nghost*2)*(ktot-nghost*2) +
	       blockIdx.z*2*(jtot-nghost*2)*(ktot-nghost*2));
  } else {
    lin_idx = (j + 
	       k*(jtot-nghost*2) + 
	       v*(jtot-nghost*2)*(ktot-nghost*2) +
	       blockIdx.z*nvar*(jtot-nghost*2)*(ktot-nghost*2));
  }

  j += nghost;
  k += nghost;

  if(j+nghost > jtot-1 or k+nghost > ktot-1) return;

  s += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;

  if(group){
    double sum=0;
    if(v==0){
      for(int vv=0; vv<nvar-1; vv++){
	sum += s[vv]*s[vv];
      }
    } else {
      sum = s[nvar-1]*s[nvar-1];
    }
    wrk[lin_idx] = sum;
  } else {
    wrk[lin_idx] = s[v]*s[v];  
  }

}

__global__ void reorder_no_square(int jtot, int ktot, int nvar, int nghost, double* s, double* wrk){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int v  = threadIdx.z;

  int lin_idx;

  lin_idx = (j + 
	     k*(jtot-nghost*2) + 
	     v*(jtot-nghost*2)*(ktot-nghost*2) +
	     blockIdx.z*nvar*(jtot-nghost*2)*(ktot-nghost*2));

  j += nghost;
  k += nghost;

  if(j+nghost > jtot-1 or k+nghost > ktot-1) return;

  s += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;

  wrk[lin_idx] = s[v];  

}


__global__ void eachsum(double* a, int n, double* b){
  
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


void G2D::l2norm(double* vec, double* l2){

  // int qcount    = nl*jtot*ktot*nvar;
  int i, leftover, smem;
  int j, k, l, v;

  dim3 vthr(32,4,nvar);
  dim3 vblk;
  vblk.x = (jtot-1-nghost*2)/vthr.x+1;
  vblk.y = (ktot-1-nghost*2)/vthr.y+1;
  vblk.z = nl;

  int pts = (jtot-nghost*2)*(ktot-nghost*2);

  int c=0;
  double* scratch1 = &wrk[c]; c+= pts*nvar*nl;
  double* scratch2 = &wrk[c]; c+= pts*nvar*nl;

  // do not group
  reorder_and_square<0><<<vblk,vthr>>>(jtot,ktot,nvar,nghost,vec,scratch1);

  dim3 threads(1,1,1), blocks(1,1,nl);

  int n     = pts*nvar;
  int power = min(9, (int)ceil(log2(n*1.0)));

  threads.x = pow(2,power);
  leftover = n;
  i = 0;
  while(leftover > 1){

    blocks.x = (leftover - 1)/ threads.x + 1;
    smem = threads.x*sizeof(double);

    if(i%2 == 0){
      eachsum<<<blocks,threads,smem>>>(scratch1, leftover, scratch2);
    } else {
      eachsum<<<blocks,threads,smem>>>(scratch2, leftover, scratch1);
    }
    i++;
    leftover = blocks.x;
  }
  
  if(i%2 == 1){
    HANDLE_ERROR(cudaMemcpy(l2,scratch2,nl*sizeof(double),cudaMemcpyDeviceToHost));
  } else {
    HANDLE_ERROR(cudaMemcpy(l2,scratch1,nl*sizeof(double),cudaMemcpyDeviceToHost));
  }

  for(l=0;l<nl;l++){
    l2[l] = sqrt(l2[l]);
  }

}

__global__ void save_q(int pts, double* q, double* qsafe, unsigned char* flags){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = blockIdx.z;
  if(i<pts){
    if(flags[l] & F_SAFE){
      // q is good, save it
      qsafe[i + l*pts] = q[i + l*pts];
    }
  }
}


void G2D::compute_residual(double* s, int isub){

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
 
#ifdef GROUP_MEANFLOW
  reorder_and_square<1><<<vblk,vthr>>>(jtot,ktot,nvar,nghost,s,scratch1);
#else
  reorder_and_square<0><<<vblk,vthr>>>(jtot,ktot,nvar,nghost,s,scratch1);
#endif

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
      eachsum<<<blocks,threads,smem>>>(scratch1, leftover, scratch2);
    } else {
      eachsum<<<blocks,threads,smem>>>(scratch2, leftover, scratch1);
    }
    i++;
    leftover = blocks.x;
  }
  
  if(i%2 == 1){
    HANDLE_ERROR(cudaMemcpy(l2var,scratch2,nv*nl*sizeof(double),cudaMemcpyDeviceToHost));
  } else {
    HANDLE_ERROR(cudaMemcpy(l2var,scratch1,nv*nl*sizeof(double),cudaMemcpyDeviceToHost));
  }

  double elapsed = timer.peek();

  bool nan_found = false;
  double l2all;

  bool first=false;
  if(not res0){
    res0  = new double[nl];
    first = true;
  }

  for(l=0; l<nl; l++){

    nan_found = false;

    if(not resfile[l]){
      // open the residual files here but leave them open
      resfile[l] = fopen(res_fname[l].c_str(), "a");
    }

    // if((nl==2 and l==1) or (nl==1)) 

    l2all=0.0;
#ifdef PRINT_STDIO
    printf("# %6d %3d %2d ", istep, isub, 0);
#endif
    fprintf(resfile[l],"%6d %3d %2d ", istep, isub, 0);
    for(v=0; v<nv; v++){
      l2all += l2var[v+l*nv];
#ifdef PRINT_STDIO
      printf("%16.8e ", sqrt(l2var[v + l*nv]));
#endif

      // nan_found = (nan_found or not isfinite(l2var[v + l*nv]));
      if(not isfinite(l2var[v + l*nv]) ){
	flags[CPU][l]   = F_TIMEACC | F_NAN | F_RECOVER; // set these flag and force timeaccuracy for subsequent steps
	l2var[v + l*nv] = BIGRES;
	l2all           = BIGRES;
	nan_found = true;
      } 

      fprintf(resfile[l],"%16.8e ", sqrt(l2var[v + l*nv]));

    }
    l2all = sqrt(l2all);
#ifdef PRINT_STDIO
    printf("%16.8e #\n", l2all);
    // if(nan_found) printf("Switching to timeaccurate: l=%d\n", l);
#endif
    fprintf(resfile[l],"%16.8e %14.6e #\n", l2all, elapsed);
    if(isub==0 or !(flags[CPU][l] & F_TIMEACC)){
      if(!nan_found and l2all < res[l]){
	// if residual decreased, set the flag to save this q vector in case of a later crash
	flags[CPU][l] = flags[CPU][l] | F_SAFE; 
	// printf("Saving Q\n");
      } else {
	// else clear that bit with bit-wise AND with bit-wise NOT of F_SAFE (1011) 
	flags[CPU][l] = flags[CPU][l] & ~F_SAFE; 
	// printf("NOT safe\n");
      }
      res[l] = l2all;
      if(first){ 
	res0[l] = l2all;
      }
    }
  }

  HANDLE_ERROR( cudaMemcpy(flags[GPU], flags[CPU], nl, cudaMemcpyHostToDevice) );

  threads  = dim3(256,1,1);
  blocks   = dim3(1,1,nl);
  blocks.x = (jtot*ktot*nvar-1)/threads.x+1;  
  save_q<<<blocks,threads>>>(jtot*ktot*nvar, q[GPU], qsafe, flags[GPU]);

  if(nan_found){
    this->debug_flag = 1;
  //   this->compute_rhs(q[GPU],s);
  //   double nrm;
  //   this->l2norm(s,&nrm);
  //   printf("norm is : %16.8e\n", nrm);
  }

  // leave the file open for writing in case we do GMRES and write linear iteration info

  // delete[] scpu;
  // delete[] scpu2;

  delete[] l2var;

  // if(nan_found) this->istep=999999999;

}


__global__ void recover_q(int pts, double* q, double* qp, double* qsafe, unsigned char* flags){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = blockIdx.z;
  if(i<pts){
    if(flags[l] & F_RECOVER){
      q[i  + l*pts]  = qsafe[i + l*pts];
      qp[i + l*pts]  = qsafe[i + l*pts];
    } 
  }
}

__global__ void unset_nan_flag(int nl, unsigned char* flags){
  int l = blockDim.x * blockIdx.x + threadIdx.x;
  if(l<nl){
    flags[l] = flags[l] & ~F_RECOVER;
  }
}

void G2D::checkpoint(){

  dim3 threads(256,1,1);
  dim3 blocks(1,1,nl);

  blocks.x  = (jtot*ktot*nvar-1)/threads.x+1;

  for(int l=0; l<nl; l++){
    if(flags[CPU][l] & F_RECOVER){
      printf("# *** Error with M=%9.3f, Alpha=%9.3f, Re=%16.8e ***\n",machs[CPU][l], aoas[CPU][l], reys[CPU][l]*machs[CPU][l]);
      printf("# *** reverting to safe solution and switching to time-accurate.\n");
    	     
    } 
    flags[CPU][l] = flags[CPU][l] & ~F_RECOVER;
  }

  recover_q<<<blocks,threads>>>(jtot*ktot*nvar, q[GPU], qp, qsafe, flags[GPU]);

  threads.x = 128;
  blocks.x  = (nl-1)/threads.x+1;
  blocks.z  = 1;
  unset_nan_flag<<<blocks,threads>>>(nl,flags[GPU]);


}
