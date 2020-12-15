#include "g2d.h"

#define WEIGHT
#define EPS 1e-15

__global__ void sa_vort(int jtot,int ktot,int nvar,int nghost,double* q, double* vorticity,
			double2* Sj,double2* Sk,double* vol){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  if(j > jtot-1-2*nghost or k > ktot-1-2*nghost) return;
  j += nghost;
  k += nghost;

  q         += (blockIdx.z*jtot*ktot)*nvar;
  vorticity += (blockIdx.z*jtot*ktot);

  int grid_idx = j + k*jtot;
  // int soln_idx = grid_idx*nvar;

  int jstride = 1;
  int kstride = jtot;

  int soln_idx_p1_j = (grid_idx-jstride)*(nvar);
  int soln_idx_p1_k = (grid_idx-kstride)*(nvar);

  int soln_idx_m1_j = (grid_idx-jstride)*(nvar);
  int soln_idx_m1_k = (grid_idx-kstride)*(nvar);

  double jac = 1.0/vol[grid_idx];

  // Average Metrics:
  double   xi_x = 0.5*(Sj[grid_idx].x + Sj[grid_idx+jstride].x)*jac;
  double   xi_y = 0.5*(Sj[grid_idx].y + Sj[grid_idx+jstride].y)*jac;
  double  eta_x = 0.5*(Sk[grid_idx].x + Sk[grid_idx+kstride].x)*jac;
  double  eta_y = 0.5*(Sk[grid_idx].y + Sk[grid_idx+kstride].y)*jac;

  //// Second Order Differences:
  double u_xi   = 0.5*(q[soln_idx_p1_j+1]/q[soln_idx_p1_j]-q[soln_idx_m1_j+1]/q[soln_idx_m1_j]);
  double v_xi   = 0.5*(q[soln_idx_p1_j+2]/q[soln_idx_p1_j]-q[soln_idx_m1_j+2]/q[soln_idx_m1_j]);
  double u_eta  = 0.5*(q[soln_idx_p1_k+1]/q[soln_idx_p1_k]-q[soln_idx_m1_k+1]/q[soln_idx_m1_k]);
  double v_eta  = 0.5*(q[soln_idx_p1_k+2]/q[soln_idx_p1_k]-q[soln_idx_m1_k+2]/q[soln_idx_m1_k]);

  double dudx = xi_x*u_xi + eta_x*u_eta;  
  double dvdy = xi_y*v_xi + eta_y*v_eta;  
  double dudy = xi_y*u_xi + eta_y*u_eta;
  double dvdx = xi_x*v_xi + eta_x*v_eta;

  double vort = sqrt((dvdx - dudy)*(dvdx - dudy));

  // vorticity[grid_idx] = vort; // plain vorticity

  double grad_av = (dudx + dvdy)/3.0; // set to 3.0 to match 3D Garfield
  double S_xx    = dudx - grad_av; 
  double S_yy    = dvdy - grad_av;
  double S_xy    = dudy + dvdx;

  double strain = sqrt(2.0*(S_xx*S_xx + S_yy*S_yy) + S_xy*S_xy);

  // Dylan: In some weird cases, vort and strain are small (~1e-28)
  // but (strain-vort) < 0. This leads to negative vorticity and
  // causes the SA model to blow up. Fix this by using a max()
  // function.
  vorticity[grid_idx] = max(vort + 2.0*min(0.0, strain - vort), 0.0);

}

__global__ void sa_vort_lls(int jtot, int ktot, int nvar, int nghost, double* q, double* vorticity, double2* xc){

  extern __shared__ double smem[];

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  if(j > jtot-1-2*nghost or k > ktot-1-2*nghost) return;
  j += nghost;
  k += nghost;

  q         += (blockIdx.z*jtot*ktot)*nvar;
  vorticity += (blockIdx.z*jtot*ktot);

  int grid_idx = j + k*jtot;

  int i, iidx;

  int gidx[5];

  // we set aside two matrices for scratch: four_1 and four_2. these
  // have 9 spots each (for 3x3 matrices) so 18 total per kernel
  double* four1 = &smem[8*(threadIdx.x + threadIdx.y*blockDim.x)];
  double* four2 = &four1[4];

  four1[0] = 0.0;
  four1[1] = 0.0;
  four1[3] = 0.0;

  // now fill the indices of the stencil we want to use for the point
  // cloud (recall we have up to 10 spots)
  gidx[0] = grid_idx;            
  gidx[1] = grid_idx - 1;
  gidx[2] = grid_idx + 1;
  gidx[3] = grid_idx - jtot;
  gidx[4] = grid_idx + jtot;

  double wi, ictwc, ctwc;
  double2 tmp1, tmp2, ctwb, bi;

  ctwb = make_double2(0.0,0.0);
  ctwc = 0.0;

  for(i=0; i<5; i++){
    iidx = gidx[i];

    bi   = xc[iidx]-xc[grid_idx];

#ifdef WEIGHT
    wi     = 1.0/((bi.x*bi.x + bi.y*bi.y)+EPS);
#else
    wi     = 1.0;
#endif
    
    ctwb.x += bi.x*wi;
    ctwb.y += bi.y*wi;
    ctwc   += wi;
    
    // this is btwb
    four1[0] += bi.x*bi.x*wi;
    four1[3] += bi.y*bi.y*wi;
    four1[1] += bi.x*bi.y*wi;
  }

  // btwb is symetric
  four1[2] = four1[1];

  // get inverse ctwc (scalar!)
  ictwc = 1.0/ctwc;

  // lhs of grad equation
  four1[0] += -ctwb.x*ictwc*ctwb.x;
  four1[1] += -ctwb.x*ictwc*ctwb.y;
  four1[3] += -ctwb.y*ictwc*ctwb.y;
  // lhs is symetric
  four1[2] = four1[1];

  // store inverse lhs in four2 temporarily
  tmp1.x = 1.0/(four1[0]*four1[3]-four1[1]*four1[2]); // determinant
  four2[0] =  four1[3]*tmp1.x;
  four2[3] =  four1[0]*tmp1.x;
  four2[1] = -four1[2]*tmp1.x;
  four2[2] = -four1[1]*tmp1.x;

  double dudx = 0.0;
  double dudy = 0.0;
  double dvdx = 0.0;
  double dvdy = 0.0;
  double irho;

  tmp1.x = 0.0;
  tmp2.x = 0.0;

  for(i=0; i<5; i++){
    iidx = gidx[i];

    bi   = xc[iidx]-xc[grid_idx];

#ifdef WEIGHT
    wi     = 1.0/((bi.x*bi.x + bi.y*bi.y)+EPS);
#else
    wi     = 1.0;
#endif

    // this only the rhs "K" matrix column
    tmp1.x = (bi.x*wi - ctwb.x*ictwc*wi);
    tmp1.y = (bi.y*wi - ctwb.y*ictwc*wi);

    // apply contribution from inverted lhs:
    tmp2.x = tmp1.x*four2[0] + tmp1.y*four2[1];
    tmp2.y = tmp1.x*four2[2] + tmp1.y*four2[3];

    irho  = 1.0/q[iidx*nvar];
    dudx += tmp2.x * q[iidx*nvar+1]*irho;
    dudy += tmp2.y * q[iidx*nvar+1]*irho;
    dvdx += tmp2.x * q[iidx*nvar+2]*irho;
    dvdy += tmp2.y * q[iidx*nvar+2]*irho;

  }

  double vort = sqrt((dvdx - dudy)*(dvdx - dudy));

  // vorticity[grid_idx] = vort; // plain vorticity

  double grad_av = (dudx + dvdy)/3.0; // set to 3.0 to match 3D Garfield
  double S_xx    = dudx - grad_av; 
  double S_yy    = dvdy - grad_av;
  double S_xy    = dudy + dvdx;

  double strain = sqrt(2.0*(S_xx*S_xx + S_yy*S_yy) + S_xy*S_xy);

  // Dylan: In some weird cases, vort and strain are small (~1e-28)
  // but (strain-vort) < 0. This leads to negative vorticity and
  // causes the SA model to blow up. Fix this by using a max()
  // function.
  vorticity[grid_idx] = max(vort + 2.0*min(0.0, strain - vort), 0.0);

}



void G2D::compute_vorticity(double* qtest, double* vort){

  int nl    = nM*nRey*nAoa;
  // int count = nl*jtot*ktot;

  dim3 thr(16,16,1);
  dim3 blk;
  blk.x = (jtot-1-nghost*2)/thr.x+1;
  blk.y = (ktot-1-nghost*2)/thr.y+1;
  blk.z = nl;

  // sa_vort<<<blk,thr>>>(jtot,ktot,nvar,nghost,qtest,vort,Sj,Sk,vol);

  size_t smem = thr.x*thr.y*8*sizeof(double);
  sa_vort_lls<<<blk,thr,smem>>>(jtot,ktot,nvar,nghost,qtest,vort,xc);

}
