#include "g2d.h"
#include <cstdio>
#include <cstdlib>

#define BIG   1e100

__global__ void shift_q(double* q_old, double* q_new, double* safe_old, double* safe_new, int tot, int* lmap){
  int i    = blockDim.x * blockIdx.x + threadIdx.x;
  int lnew = blockIdx.z;
  int lold = lmap[lnew];
  if(i<tot){
    q_new[   i+lnew*tot] = q_old[   i+lold*tot];
    safe_new[i+lnew*tot] = safe_old[i+lold*tot];
  }
}


void G2D::check_convergence(){

  int l;
  bool* done = new bool[nl];

  double drop;

  double fmin, fmax, fvary;

  double eps=1e-16;
  double r0;

  for(l=0; l<nl; l++){ 
    // assume we're not done
    done[l] = false;

    if(res0) r0 = res0[l];
    else     r0 = res[l];

    drop = log10(r0/res[l]);

    fmin =  BIG;
    fmax = -BIG;
    for(int i=0; i<AVG_HIST; i++){
      fmin = std::min(fmin, fhist[l*AVG_HIST+i]);
      fmax = std::max(fmax, fhist[l*AVG_HIST+i]);
    }
    fvary = 200*(fmax-fmin)/(fmax+fmin+eps);

    // printf("# CASE : M=%9.3f, Alpha=%9.3f, Re=%16.8e, drop=%7.3f, f_vary=%8.3f\n",  
    // 	   machs[CPU][l], aoas[CPU][l], reys[CPU][l]*machs[CPU][l], drop, fvary);

    double re = (vary_Re_with_Mach)? reys[CPU][l] : reys[CPU][l]*machs[CPU][l];

    // First criteria: residual converges more than 6 orders
    if(drop > 6){
      if(flags[CPU][l] & F_TIMEACC){
	done[l] = true;
	printf("# [%16s] DONE : M=%9.3f, Alpha=%9.3f, Re=%16.8e, dropped 5 orders\n", 
	       foilname.c_str(),machs[CPU][l], aoas[CPU][l], re);
      } else {
	flags[CPU][l] = flags[CPU][l] | F_TIMEACC;
	printf("# [%16s] Final Stretch : M=%9.3f, Alpha=%9.3f, Re=%16.8e, dropped 5 orders\n", 
	       foilname.c_str(),machs[CPU][l], aoas[CPU][l], re);
      }
      continue;      
    }

    // Second criteria: forces have not changed more than 0.1%
    if(fvary < 0.1 and drop > 5){
      if(flags[CPU][l] & F_TIMEACC){      
	done[l] = true;
	printf("# [%16s] DONE : M=%9.3f, Alpha=%9.3f, Re=%16.8e, <0.1%% change in forces\n", 
	       foilname.c_str(),machs[CPU][l], aoas[CPU][l], re);
      }	else {
	flags[CPU][l] = flags[CPU][l] | F_TIMEACC;	
	printf("# [%16s] Final Stretch : M=%9.3f, Alpha=%9.3f, Re=%16.8e, <0.1%% change in forces\n", 
	       foilname.c_str(),machs[CPU][l], aoas[CPU][l], re);
      }
      continue;
    }

  }

  int* lmap = new int[nl];

  int ll=0;
  for(l=0; l<nl; l++){

    if(done[l]){
      // continue without incrementing ll ( and start shifting next time )
      continue;
    }

    lmap[ll] = l;

    if(l == ll){
      // we haven't hit any completed cases, increment ll and move on
      ll++;
      continue;
    }

    // we need to shift:
    flags[CPU][ll]   = flags[CPU][l];
    machs[CPU][ll]   = machs[CPU][l];
    aoas[CPU][ll]    = aoas[CPU][l];
    reys[CPU][ll]    = reys[CPU][l];

    res_fname[ll]    = res_fname[l];
    forces_fname[ll] = forces_fname[l];
    cpcf_fname[ll]   = cpcf_fname[l];
    sol_fname[ll]    = sol_fname[l];

    res[ll]          = res[l];
    if(res0){
      res0[ll]         = res0[l];
    }

    ll++;

  }

  this->nl = ll;

  int*    lmap_gpu = (int*)wrk;
  double* qtmp     = &wrk[nl];
  int qcount       = nl*jtot*ktot*nvar;

  HANDLE_ERROR( cudaMemcpy(qtmp, qsafe, qcount*sizeof(double), cudaMemcpyDeviceToDevice) );

  // copy the map to the GPU
  HANDLE_ERROR( cudaMemcpy(lmap_gpu, lmap, nl*sizeof(int), cudaMemcpyHostToDevice) );
  
  // copy the shifted M, Aoa, Rey values to the GPU (small data so it's ok)
  HANDLE_ERROR( cudaMemcpy(this->machs[GPU], this->machs[CPU], nl*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->aoas[GPU],  this->aoas[CPU],  nl*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->reys[GPU],  this->reys[CPU],  nl*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->flags[GPU], this->flags[CPU], nl,                cudaMemcpyHostToDevice) );
  
  double* newq = this->s;
  
  dim3 thr(256,1,1);
  dim3 blk(1,1,nl);
  blk.x = (jtot*ktot*nvar-1)/thr.x+1;
  
  // Shift q into "newq" (which is s residual storage). 
  shift_q<<<blk,thr>>>(q[GPU],newq,qtmp,qsafe,jtot*ktot*nvar,lmap_gpu);

  // Then swap s and q pointers.
  this->s      = this->q[GPU];
  this->q[GPU] = newq;

  delete[] done;
  delete[] lmap;

}
