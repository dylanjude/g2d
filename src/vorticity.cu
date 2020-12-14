#include "g2d.h"


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


void G2D::compute_vorticity(double* qtest, double* vort){

  int nl    = nM*nRey*nAoa;
  int count = nl*jtot*ktot;

  dim3 thr(16,16,1);
  dim3 blk;
  blk.x = (jtot-1-nghost*2)/thr.x+1;
  blk.y = (ktot-1-nghost*2)/thr.y+1;
  blk.z = nl;

  sa_vort<<<blk,thr>>>(jtot,ktot,nvar,nghost,qtest,vort,Sj,Sk,vol);

}
