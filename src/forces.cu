#include "g2d.h"

#define INTEGRATE 0
#define CPCF      1

template<int mode>
__global__ void wall_forces(int jtot, int ktot, int nvar, int nghost, double* q, double* mulam, double* f, 
			    double2* Sj, double2* Sk, double2* x, double* vol,
			    bool visc, double* reys){
  
  int j,k;
  
  j   = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  k   = nghost;

  if(j+nghost > jtot-1) return;

  double rey = reys[blockIdx.z];

  q     += (j + k*jtot + blockIdx.z*ktot*jtot)*nvar;
  if(mode==CPCF){
    f     += j-nghost    + blockIdx.z*(jtot-nghost*2)*2; // from fast to slow: j, v, l, where v is [0,1]
  } else {
    f     += j-nghost    + blockIdx.z*(jtot-nghost*2)*3; // from fast to slow: j, v, l, where v is [0,1,2]
  }
  mulam += j + k*jtot  + blockIdx.z*ktot*jtot;
  
  double u  = q[1]/q[0];
  double v  = q[2]/q[0];
  double p1 = (GAMMA - 1.0)*(q[3] - 0.5*(q[1]*q[1] + q[2]*q[2])/q[0]);
  
  q += nvar*jtot; // step in k-direction
  // double du = q[1]/q[0] - u;
  // double dv = q[2]/q[0] - v;

  double du = 2*u;
  double dv = 2*v;

  double p2 = (GAMMA - 1.0)*(q[3] - 0.5*(q[1]*q[1] + q[2]*q[2])/q[0]);

  double p_wall = 0.5*(3.0*p1-p2); // extrapolate pressure to wall

  double2 n, t;
  double idn;

  // average the Sj vector to get the distance from the wall to the
  // center of the cell
  // n = 0.5*(Sj[j+k*jtot]+Sj[j+1+k*jtot]);
  // idn = 1.0/sqrt(dot(n,n));

  // Now use the Sk vector as the wall normal scaled by the "area"
  n   = Sk[j+k*jtot];
  t.x = n.y;
  t.y = -n.x; // the tangent vector is the inverse slope of n

  double area = sqrt(dot(n,n));
  t = t/area;
  n = n/area;

  idn = area/vol[j+k*jtot];

  double mu, f_wall=0.0;

  if(visc){
    mu = 1.0/rey;
    mu = mu*0.5*(3*mulam[0] - mulam[jtot]);
    // mu = mu*0.5*(mulam[0] + mulam[jtot]);
    // mu = mu*mulam[0];
    // du is 0.5*u because no grid motion
    f_wall = mu*(du*t.x + dv*t.y)*idn;
  }

  double2 ff = (f_wall*t - p_wall*n)*area;        // force at panel center
  double2 r  = 0.5*(x[j+k*jtot] + x[j+1+k*jtot]); // coord of panel center

  r.x = -(r.x-0.25); // vector from the panel TO the quarter
  r.y = -r.y;        // chord. Then r cross ff gives a nose-up moment.

  // these are strided for easy summation later
  if(mode==CPCF){
    f[                0] = p_wall;                     // x-force
    f[  (jtot-nghost*2)] = f_wall;                     // y-force
  } else {
    f[                0] = ff.x;                      // x-force
    f[  (jtot-nghost*2)] = ff.y;                      // y-force
    f[2*(jtot-nghost*2)] = r.x*ff.y - r.y*ff.x;       // z-moment
  }

}

__global__ void sum3(double* a, int n, double* b){
  
  extern __shared__ double ish1[];
  int tid = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int v = blockIdx.y; // which var to sum: fx, fy or mz
  int l = blockIdx.z; // which airfoil

  // initially there are n values between each variable on each grid
  if(i<n){
    ish1[tid] = a[i + v*n + l*n*gridDim.y]; // gridDim.y is nvar, which here is 3
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


void G2D::check_forces(){

  dim3 thr(32,1,1);
  dim3 blk(1,1,nl);
  blk.x     = (jtot-1-nghost*2)/thr.x+1; // only j-points, no ghosts

  int c=0;
  double* f1 = &wrk[c]; c+= 3*(jtot-nghost*2)*nl;
  double* f2 = &wrk[c]; c+= 3*(jtot-nghost*2)*nl;

  double* fsum  = new double[3*nl];

  FILE* fid;

  bool visc = (this->eqns != EULER);

  wall_forces<INTEGRATE><<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],mulam,f1,Sj,Sk,x[GPU],vol,visc,reys[GPU]);

  dim3 threads(1,1,1), blocks(1,3,nl); // because 3 variables: fx, fy, mz

  int n     = jtot-nghost*2;
  int power = min(9, (int)ceil(log2(n*1.0)));
  size_t smem;

  threads.x = pow(2,power);
  int leftover = n;
  int i = 0;
  while(leftover > 1){

    blocks.x = (leftover - 1)/ threads.x + 1;
    smem = threads.x*sizeof(double);

    if(i%2 == 0){
      sum3<<<blocks,threads,smem>>>(f1, leftover, f2);
    } else {
      sum3<<<blocks,threads,smem>>>(f2, leftover, f1);
    }
    i++;
    leftover = blocks.x;
  }
  
  if(i%2 == 1){
    HANDLE_ERROR(cudaMemcpy(fsum,f2,3*nl*sizeof(double),cudaMemcpyDeviceToHost));
  } else {
    HANDLE_ERROR(cudaMemcpy(fsum,f1,3*nl*sizeof(double),cudaMemcpyDeviceToHost));
  }

  double Cl, Cd, Cm;
  double Cn, Cc, alpha, fac;

  double fsave;

  for(int l=0; l<nl; l++){

    fid = fopen(forces_fname[l].c_str(),"a");

    fac = 1.0/(0.5*machs[CPU][l]*machs[CPU][l]); // 1.0 / (0.5 * rho_inf * v_inf**2)

    alpha = aoas[CPU][l]*PI/180;

    Cc = fac*fsum[l*3];
    Cn = fac*fsum[l*3+1];
    Cm = fac*fsum[l*3+2];
	  
    Cl = Cn*cos(alpha) - Cc*sin(alpha);
    Cd = Cc*cos(alpha) + Cn*sin(alpha);

    fhist[l*AVG_HIST+(iforce%AVG_HIST)] = Cl + 10*Cd + 100*Cm;

    // printf("alpha: %f, Mach : %f\n", alpha, machs[CPU][im]);

    fprintf(fid, "%16.8e %16.8e %16.8e\n", Cl, Cd, Cm);
    // printf("__forces__  %16.8e %16.8e %16.8e\n", Cl, Cd, Cm);

    fclose(fid);
  }

  delete[] fsum;

  iforce++;

}


void G2D::write_cpcf(){

  dim3 thr(32,1,1);
  dim3 blk(1,1,nl);
  blk.x     = (jtot-1-nghost*2)/thr.x+1; // only j-points, no ghosts

  int c=0;
  double* f1 = &wrk[c]; c+= 2*(jtot-nghost*2)*nl;

  FILE* fid;

  bool visc = (this->eqns != EULER);

  wall_forces<CPCF><<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],mulam,f1,Sj,Sk,x[GPU],vol,visc,reys[GPU]);

  double* fcpu0 = new double[2*(jtot-nghost*2)*nl];
  double* fcpu  = fcpu0;
  HANDLE_ERROR(cudaMemcpy(fcpu,f1,2*(jtot-nghost*2)*nl*sizeof(double),cudaMemcpyDeviceToHost));

  for(int l=0; l<nl; l++){

    fid = fopen(cpcf_fname[l].c_str(),"w"); // re-write the file each time

    fcpu = &fcpu0[l*2*(jtot-nghost*2)];

    for(int j=0; j<jtot-nghost*2; j++){

      double xx = 0.5*(x[CPU][j+nghost+nghost*jtot].x + x[CPU][j+nghost+1+nghost*jtot].x);
      double p  = (fcpu[j + (jtot-nghost*2)*0] - 1.0/GAMMA)/(0.5*machs[CPU][l]*machs[CPU][l]);
      // double p  = (fcpu[j + (jtot-nghost*2)*0])/(0.5*machs[CPU][l]*machs[CPU][l]);
      double f  = (fcpu[j + (jtot-nghost*2)*1]            )/(0.5*machs[CPU][l]*machs[CPU][l]);
      
      fprintf(fid, "%16.8e %16.8e %16.8e\n", xx, p, f);
    }

    fclose(fid);

  }

  delete[] fcpu0;

}
