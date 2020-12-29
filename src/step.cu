#include "g2d.h"

__global__ void set_dt(int jtot,int ktot,int nvar,int nghost,
		       double* q, double* dt, double* vol, double2* Sj, double2* Sk, double cfl){

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

  // double eigmax = abs(uu) + xsc + abs(vv) + ysc;
  double eigmax = abs(uu) + xsc + abs(vv) + ysc + sqrt(c2*vol[gidx]*vol[gidx]); // <-- last term to match 3d Garfield

  // int ib = (j+nghost < jtot and k+nghost < ktot);
  int ib = 1;//(j+nghost < jtot and k+nghost < ktot);

  dt[0] = ib*vol[gidx]*cfl/eigmax;

}

#define DBGJ 191
#define DBGK 50

__global__ void update_q(int jtot,int ktot,int nvar,int nghost, double* q, double* s){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;

  q  += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  s  += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;

  if(j+nghost < jtot and k+nghost < ktot){

    for(int v=0; v<nvar; v++){
      q[v] += s[v];
    }
  }

}

void G2D::go(){

  double cfl = 100.0;

  this->take_steps(100,1,cfl);
  this->take_steps(100,1,cfl);
  this->take_steps(100,1,cfl);
  this->take_steps(100,1,cfl);

}


void G2D::take_steps(int nstep, int nsub, double cfl0){

  int checkmod=10;

  int qcount = nl*jtot*ktot*nvar;

  dim3 thr(16,16,1);
  dim3 blk;
  blk.x = (jtot-1)/thr.x+1;
  blk.y = (ktot-1)/thr.y+1;
  blk.z = nl;

  double cfl  = cfl0;

  bool check;

  int nl0 = nM*nAoa*nRey;

  for(int i=0; i<nstep && istep<1000000 && nl>0; i++){

    HANDLE_ERROR( cudaMemcpy(qp, q[GPU], qcount*sizeof(double), cudaMemcpyDeviceToDevice) );

    // CFL Ramping (linear)
    if(istep < 100){
      cfl = 1.0 + (cfl0-1)*istep/100;
    } else {
      cfl = cfl0;
    }

    check = ((i+1) % checkmod == 0);

    set_dt<<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],dt,vol,Sj,Sk,cfl);

    for(int isub=0; isub<nsub; isub++){

      this->istep++; // increment global step count

      this->compute_rhs(q[GPU],s);

      if(check){
	this->compute_residual(s, isub);
      }

      // this->precondition(s,s);
      this->gmres(s);

      update_q<<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],s);

    }

    if(check){

      printf("Iteration %6d : running %4d conditions (%4d complete)\n", istep, nl, nl0-nl);

      // monitor forces
      this->check_forces();

      // close the residual files (after gmres runs, since gmres also prints to that file)
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
