#include "g2d.h"

__global__ void set_dt(int jtot,int ktot,int nvar,int nghost, double* q, double* dt, double* vol, 
		       double2* Sj, double2* Sk, double cfl, unsigned char* flags, double* machs){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  if(j>jtot-1 or k>ktot-1) return;

  q   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  dt  += j      + k*jtot      + blockIdx.z*jtot*ktot;

  double irho = 1.0/q[0];
  double u    = q[1]*irho;
  double v    = q[2]*irho;
  double p    = (GAMMA - 1.0)*(q[3] - 0.5*q[0]*(u*u + v*v)); 
  double c2   = GAMMA*p*irho;

  int gidx = j + k*jtot;

  double uu   = u*Sj[gidx].x + v*Sj[gidx].y;
  double vv   = u*Sk[gidx].x + v*Sk[gidx].y;

  double xs2  = Sj[gidx].x*Sj[gidx].x + Sj[gidx].y*Sj[gidx].y;
  double ys2  = Sk[gidx].x*Sk[gidx].x + Sk[gidx].y*Sk[gidx].y;

  double xsc  = sqrt(c2*xs2);
  double ysc  = sqrt(c2*ys2);

  double eigmax = abs(uu) + xsc + abs(vv) + ysc;
  // double eigmax = abs(uu) + xsc + abs(vv) + ysc + sqrt(c2*vol[gidx]*vol[gidx]); // <-- last term to match 3d Garfield

  // int ib = (j+nghost < jtot and k+nghost < ktot);
  int ib = 1;//(j+nghost < jtot and k+nghost < ktot);

  double dt_local  = ib*vol[gidx]*cfl/eigmax;

  // if we got a NaN recently, then take a smaller timestep
  // double safe_fac = (flags[blockIdx.z] & F_NAN)? 0.5 : 1.0;
  double safe_fac = 1;

  // Check if timeaccurate using bit-wise AND of flags
  if(flags[blockIdx.z] & F_TIMEACC){
    double M = machs[blockIdx.z];
    double dt_global = DT_GLOBAL(M)*safe_fac;
    // dt_local *= 10;
    dt[0] = dt_global / (1.0 + dt_global/dt_local);
  } else {
    dt[0] = dt_local;
  }

}

#define DBGJ 191
#define DBGK 50

__global__ void update_q(int jtot,int ktot,int nvar,int nghost, double* q, double* s){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;
  int v  = threadIdx.z;

  q  += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  s  += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;

  if(j+nghost < jtot and k+nghost < ktot){
    q[v] += s[v];
  }

}

__global__ void save_qp(int jtot,int ktot,int nvar, double* q, double* qp,unsigned char* flags,bool first_sub){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int v  = threadIdx.z;

  if(j > jtot-1 or k > ktot-1) return;

  q   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  qp  += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;

  // if(j==0 and k==0 and v==0){
  //   printf("timeacc ? %d \n", bool(flags[blockIdx.z] & F_TIMEACC));
  // }

  // if we're not timeaccurate or this is the first sub-iteration
  // note: flags are checked with bit-wise AND
  if(!(flags[blockIdx.z] & F_TIMEACC) or first_sub){ 
    qp[v] = q[v];
  }

}

void G2D::go(){

  double cfl;

  cfl = 100.0;
  for(int i=0; i<3; i++){
    this->take_steps(200,3,cfl);
  }

  cfl = 50.0;
  for(int i=0; i<2; i++){
    this->take_steps(200,4,cfl);
  }

  // If cases haven't converged yet, force them all to be time-accurate.
  this->all_timeacc = true;

  cfl = 100.0;
  for(int i=0; i<30; i++){
    this->take_steps(200,4,cfl);
  }

  for(int i=0; i<25; i++){
    this->take_steps(200,5,cfl);
  }

}

void G2D::take_steps(int nstep, int nsub, double cfl0){

  // int checkmod=int(10/nsub);
  int checkmod=10;

  dim3 thr(16,16,1);
  dim3 vthr(16,4,nvar);
  dim3 blk,vblk;
  blk.x  = (jtot-1)/thr.x+1;
  blk.y  = (ktot-1)/thr.y+1;
  blk.z  = nl;
  vblk.x = (jtot-1)/vthr.x+1;
  vblk.y = (ktot-1)/vthr.y+1;
  vblk.z = nl;

  double cfl  = cfl0;

  bool check;

  int nl0 = nM*nAoa*nRey;

  // Reset to just timeaccuracy flag
  if(this->all_timeacc){
    for(int l=0; l<nl; l++) flags[CPU][l] = F_TIMEACC; // force all
  } else {
    for(int l=0; l<nl; l++) flags[CPU][l] = (flags[CPU][l] & F_TIMEACC); // recover previosly set
  }
  HANDLE_ERROR( cudaMemcpy(flags[GPU], flags[CPU], nl, cudaMemcpyHostToDevice) );

  for(int i=0; i<nstep && istep<1000000 && nl>0; i++){

    check = ((i+1) % checkmod == 0);

    // CFL Ramping (linear)
    if(istep < 100){
      cfl = 1.0 + (cfl0-1)*istep/100;
    } else {
      cfl = cfl0;
    }

    this->istep++; // increment global step count

    // some planes will be time-accurate, others not, so pass in
    // info for the kernel to figure out dt and saving qp
    set_dt<<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],dt,vol,Sj,Sk,cfl,flags[GPU],machs[GPU]);

    for(int isub=0; isub<nsub; isub++){

      save_qp<<<vblk,vthr>>>(jtot,ktot,nvar, q[GPU], qp, flags[GPU], (isub==0));

      this->compute_rhs(q[GPU],s);

      if(check){
	this->compute_residual(s, isub);
      }

      // this->precondition(s,s);
      this->gmres(s,isub);

      update_q<<<vblk,vthr>>>(jtot,ktot,nvar,nghost,q[GPU],s);

    }

    if(check){

      // If any NaNs occurred, revert to a safe Q vector
      this->checkpoint();

      printf("[%16s] %6d : running %4d conditions (%4d complete)\n", foilname.c_str(), istep, nl, nl0-nl);

      // Monitor forces
      this->check_forces();

      // Close the residual files (after gmres runs, since gmres also prints to that file)
      for(int l=0; l<nl; l++){
        if(this->resfile[l]){
          fclose(this->resfile[l]);
          this->resfile[l]=NULL;
        }
      }
    }
    
    // done timestep loop
  }

  this->write_cpcf();
  this->write_sols();

  // monitor convergence
  this->check_convergence();

}
