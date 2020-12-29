#include "g2d.h"

__global__ void do_ax(double* a, double* x, double* out, int tot){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = blockIdx.z;
  if(i<tot) out[i+l*tot] = a[l]*x[i+l*tot];
}

__global__ void do_axpy(double* a, double* x, double* y, double* out, int tot){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = blockIdx.z;
  if(i<tot) out[i+l*tot] = a[l]*x[i+l*tot] + y[i+l*tot];
}

void G2D::axpy(double* a, double* x, double* y, double* out){

  double* a_gpu = wrk;

  HANDLE_ERROR( cudaMemcpy(a_gpu, a, nl*sizeof(double), cudaMemcpyHostToDevice) );

  dim3 thr(256,1,1);
  dim3 blk(1,1,nl);
  blk.x = (jtot*ktot*nvar-1)/thr.x+1;

  if(y){
    do_axpy<<<blk,thr>>>(a_gpu, x, y, out, jtot*ktot*nvar);
  } else {
    do_ax<<<blk,thr>>>(a_gpu, x, out, jtot*ktot*nvar);
  }

}

__global__ void suml(double* a, int n, double* b){
  
  extern __shared__ double ish1[];
  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = blockIdx.z;

  if(i<n){
    ish1[tid] = a[i + l*n]; // gridDim.y is nvar
  } else {
    ish1[tid] = 0;
  }

  __syncthreads();

  for(int s=blockDim.x/2; s>0; s>>=1){
    if(tid < s){
      ish1[tid] += ish1[tid+s];
    }
    __syncthreads();
  }

  if(tid == 0){ 
    b[blockIdx.x + l*gridDim.x] = ish1[0];
  } 

}


__global__ void vec_mult(int jtot, int ktot, int nvar, int nghost, double* a, double* b, double* scr){

  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int v = threadIdx.z;
  int l = blockIdx.z;

  int lin_idx = (j + 
		 k*(jtot-nghost*2) + 
		 v*(jtot-nghost*2)*(ktot-nghost*2) +
		 blockIdx.z*nvar*(jtot-nghost*2)*(ktot-nghost*2));

  j += nghost;
  k += nghost;

  if(j+nghost > jtot-1 or k+nghost > ktot-1) return;

  a += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;
  b += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;

  scr[lin_idx] = a[v]*b[v];

}

void G2D::vdp(double* a, double* b, double* out){

  int i, leftover, smem;
  int j, k, l, v;

  dim3 thr(16,4,nvar);
  dim3 blk(1,1,nl);
  blk.x = (jtot-2*nghost-1)/thr.x+1;
  blk.y = (ktot-2*nghost-1)/thr.y+1;

  int pts  = jtot*ktot;
  int ipts = (jtot-2*nghost)*(ktot-2*nghost);

  int c=0;
  double* scratch1 = &wrk[c]; c+= ipts*nvar*nl;
  double* scratch2 = &wrk[c]; c+= ipts*nvar*nl;

  vec_mult<<<blk,thr>>>(jtot,ktot,nvar,nghost,a,b,scratch1);

  dim3 threads(1,1,1), blocks(1,1,nl);

  int n     = ipts*nvar;
  int power = min(9, (int)ceil(log2(n*1.0)));

  threads.x = pow(2,power);
  leftover = n;
  i = 0;
  while(leftover > 1){
    blocks.x = (leftover - 1)/ threads.x + 1;
    smem = threads.x*sizeof(double);
    if(i%2 == 0){
      suml<<<blocks,threads,smem>>>(scratch1, leftover, scratch2);
    } else {
      suml<<<blocks,threads,smem>>>(scratch2, leftover, scratch1);
    }
    i++;
    leftover = blocks.x;
  }
  
  if(i%2 == 1){
    HANDLE_ERROR(cudaMemcpy(out,scratch2,nl*sizeof(double),cudaMemcpyDeviceToHost));
  } else {
    HANDLE_ERROR(cudaMemcpy(out,scratch1,nl*sizeof(double),cudaMemcpyDeviceToHost));
  }

}

__global__ void plus_eps(double* qeps, double* q, double eps, double* dq, int tot){
  int i   = blockDim.x * blockIdx.x + threadIdx.x;
  int l   = blockIdx.z;
  if(i<tot) qeps[i+l*tot] = q[i+l*tot]+eps*dq[i+l*tot];
}

__global__ void minus_mult(double *rhs, double *res, double over_eps, int tot){
  int i   = blockDim.x * blockIdx.x + threadIdx.x;
  int l   = blockIdx.z;
  if(i<tot) res[i+l*tot] = (rhs[i+l*tot]-res[i+l*tot])*over_eps;
}

__global__ void time_diag(double* res, double* dq, double* dt, int pts, int nvar){

  int i   = blockDim.x * blockIdx.x + threadIdx.x;
  int l   = blockIdx.z;
  if(i<pts*nvar){ 
    double scale = (i%nvar==4)? SASCALE : 1.0;
    double idtau = scale/dt[i/nvar+l*pts];
    res[i+l*pts*nvar] += dq[i+l*pts*nvar]*idtau;
  }

}


void G2D::mvp(double* dq, double* res){

  dim3 thr(256,1,1);
  dim3 blk(1,1,nl);
  blk.x = (jtot*ktot*nvar-1)/thr.x+1;

  double* qeps = gmres_scr; // scratch mem unused by anything in compute_rhs (so not wrk)

  double eps = 1e-6;
  double inv_eps = 1.0/eps;

  double tmp;

  plus_eps<<<blk,thr>>>(qeps,this->q[GPU],eps,dq,jtot*ktot*nvar);

  // debug_print(87,2,0,qeps,5);
  // printf("-----marker\n");

  this->compute_rhs(qeps,res);

  // debug_print(77,36,0,res,5);

  // debug_print(97,2,0,this->s,5);
  // debug_print(97,2,0,res,5);

  minus_mult<<<blk,thr>>>(this->s,res,inv_eps,jtot*ktot*nvar);

  // this->l2norm(this->s,&tmp);
  // printf("__re0__ %24.16e\n",tmp);

  // this->l2norm(res,&tmp);
  // printf("__mvp__ %24.16e\n",tmp);

  // debug_print(97,2,0,res,5);

  // int alldof  = jtot*ktot*nvar*nl;
  // double* dbg = new double[alldof];     
  // FILE* DFILE = fopen("dbgdata.dat", "w");      
  // int idx,j,k,l=3;       
  // double* dbg3; 
  // HANDLE_ERROR(cudaMemcpy(dbg, res, alldof*sizeof(double), cudaMemcpyDeviceToHost));   
  // for(k=0; k<ktot; k++){ 
  //   for(j=0; j<jtot; j++){      
  //     dbg3 = &dbg[(j+k*jtot)*5];     
  //     fprintf(DFILE,"%24.16e %24.16e %24.16e %24.16e \n", dbg3[0],dbg3[1],dbg3[2],dbg3[3]);     
  //   }    
  // }      
  // fclose(DFILE);       
  // delete[] dbg; 

  // printf("----back to gmres\n");

  time_diag<<<blk,thr>>>(res,dq,dt,jtot*ktot,nvar);

  this->zero_bc(res);

  // debug_print(193,8,0,res,5);
  // debug_print(193,8,0,dt,1);

  // double dbg;
  // this->l2norm(res,&dbg);
  // if(isfinite(dbg)) return;

  // int alldof = nl*jtot*ktot*nvar;
  // double* cdbg = new double[alldof];
  // HANDLE_ERROR(cudaMemcpy(cdbg, res, alldof*sizeof(double), cudaMemcpyDeviceToHost));
  // int i=0;
  // for(int l=0; l<nl; l++){
  //   for(int k=0; k<ktot; k++){
  //     for(int j=0; j<jtot; j++){
  // 	for(int v=0; v<nvar; v++){
  // 	  if(not isfinite(cdbg[i])){
  // 	    printf("nan at %d %d %d\n", j, k, v);
  // 	  }
  // 	  i++;
  // 	}
  //     }
  //   }
  // }
  // delete[] cdbg;

  // debug_print(87,2,0,res,5);

}
