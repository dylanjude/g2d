#include "g2d.h"

#define DBGJ 87
#define DBGK 2

// #define BORGES_WEIGHT
// #define UPWIND_OPT 

#define EPSROE 1.0e-1

template<int dir>
__global__ void roe_flux(int jtot, int ktot, int nvar, int nghost,
			 double* q_l, double* q_r, double* flx, double2* Sj){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;

  q_l += (j + k*jtot + blockIdx.z*jtot*ktot)*4;
  q_r += (j + k*jtot + blockIdx.z*jtot*ktot)*4;
  flx += (j + k*jtot + blockIdx.z*jtot*ktot)*4;

  if(j+nghost > jtot or k+nghost > ktot) return; // note we want the last face, where k+nghost==ktot

  double lambda_tilda, tmp;
  double3 lam_l,lam_r,lam_av;
  double k_x, k_y;

  k_x = Sj[j+k*jtot].x;
  k_y = Sj[j+k*jtot].y;


  double rho_l  = q_l[0]; 
  double rho_r  = q_r[0];
  double rho_av = sqrt(rho_l*rho_r);
  double inv_rho_l  = 1.0/rho_l;
  double inv_rho_r  = 1.0/rho_r;

  double roe_wt_1 = sqrt(rho_l)/(sqrt(rho_l)+sqrt(rho_r));
  double roe_wt_2 = 1-roe_wt_1;

  double u_l  = q_l[1];
  double u_r  = q_r[1];
  double u_av = roe_wt_1*u_l + roe_wt_2*u_r;    

  // if(dir==0 and j==DBGJ and k==DBGK){
  //   printf("_roe_ %d %d -- %24.16e %24.16e %24.16e\n", j,k, q_l[1], inv_rho_l, u_l);
  // }

  double v_l  = q_l[2];
  double v_r  = q_r[2];
  double v_av = roe_wt_1*v_l + roe_wt_2*v_r;     

  double p_l  = q_l[3];
  double p_r  = q_r[3];

  tmp = 1.0/(GAMMA-1.0);
  double e_l  = p_l*tmp + 0.5*rho_l*(u_l*u_l + v_l*v_l);
  double e_r  = p_r*tmp + 0.5*rho_r*(u_r*u_r + v_r*v_r);

  double q_sqr_av = u_av*u_av + v_av*v_av;

  double h_l  = (e_l + p_l)*inv_rho_l;
  double h_r  = (e_r + p_r)*inv_rho_r;
  double h_av = roe_wt_1*h_l + roe_wt_2*h_r;     

  double c_l  = sqrt((GAMMA*p_l)*inv_rho_l);
  double c_r  = sqrt((GAMMA*p_r)*inv_rho_r);
  double c_av = sqrt((GAMMA-1.0)*(h_av - 0.5*(q_sqr_av))); 
  double inv_c_av = 1.0/c_av; 

  double nx = k_x;
  double ny = k_y;

  double n2 = nx*nx+ny*ny;
  double mag = sqrt(n2);
  double imag = 1.0/mag;
  double r1 = nx*imag;
  double r2 = ny*imag;

  // need vn grid here for lambdas below
  double V_l  = u_l*r1  + v_l*r2;
  double V_r  = u_r*r1  + v_r*r2;
  double V    = u_av*r1 + v_av*r2; 
  double V_av = V;

  ///////////////////////////////////////
  //////Computing Local Eigenvalues//////
  ///////////////////////////////////////
  lam_av.x = V_av; 
  lam_av.y = V_av + c_av; 
  lam_av.z = V_av - c_av; 

  lam_l.x = V_l; 
  lam_l.y = V_l + c_l; 
  lam_l.z = V_l - c_l; 

  lam_r.x = V_r; 
  lam_r.y = V_r + c_r; 
  lam_r.z = V_r - c_r; 

  double w1, w2;

  lambda_tilda = max(4.0*(lam_r.x - lam_l.x),EPSROE);
  w2 = 0.5*(1.0 + copysign(1.0,abs(lam_av.x) - 0.5*lambda_tilda));
  w1 = 1.0 - w2;
  lam_av.x = w1*(lam_av.x*lam_av.x/lambda_tilda + 
		 0.25*lambda_tilda ) + w2*( abs(lam_av.x) );
  lambda_tilda = max(4.0*(lam_r.y - lam_l.y),EPSROE);
  w2 = 0.5*(1.0 + copysign(1.0,abs(lam_av.y) - 0.5*lambda_tilda));
  w1 = 1.0 - w2;
  lam_av.y = w1*(lam_av.y*lam_av.y/lambda_tilda + 
		 0.25*lambda_tilda ) + w2*( abs(lam_av.y) );
  lambda_tilda = max(4.0*(lam_r.z - lam_l.z),EPSROE);
  w2 = 0.5*(1.0 + copysign(1.0,abs(lam_av.z) - 0.5*lambda_tilda));
  w1 = 1.0 - w2;
  lam_av.z = w1*(lam_av.z*lam_av.z/lambda_tilda + 
		 0.25*lambda_tilda ) + w2*( abs(lam_av.z) );

  double drho = rho_r - rho_l;
  double dp   = p_r - p_l;
  double du   = u_r - u_l;
  double dv   = v_r - v_l;
  double dV   = V_r - V_l;

  // quantity in parentheses blazek eq 4.90
  double drho_dp  = drho - dp*inv_c_av*inv_c_av;
  // quantity in parentheses blazek eq 4.89, 4.91
  tmp = 1.0 /(2.0*c_av*c_av);
  double dp_p_rho_c = (dp + rho_av*c_av*dV)*tmp;
  double dp_m_rho_c = (dp - rho_av*c_av*dV)*tmp;

  double dF0, dF1, dF2, dF3; // delta F, blazek eqn. 4.89-4.91

  // eqn. 4.90
  dF0 = lam_av.x * (drho_dp);
  dF1 = lam_av.x * ( (drho_dp)*u_av         + rho_av*(du - dV*r1) );
  dF2 = lam_av.x * ( (drho_dp)*v_av         + rho_av*(dv - dV*r2) );
  dF3 = lam_av.x * ( (drho_dp)*q_sqr_av*0.5 + rho_av*(u_av*du + v_av*dv - V*dV ) );

  // eqn. 4.91
  dF0 = dF0 + lam_av.y * dp_p_rho_c;
  dF1 = dF1 + lam_av.y * dp_p_rho_c * (u_av + c_av*r1);
  dF2 = dF2 + lam_av.y * dp_p_rho_c * (v_av + c_av*r2);
  dF3 = dF3 + lam_av.y * dp_p_rho_c * (h_av + c_av*V);

  // eqn 4.89
  dF0 = dF0 + lam_av.z * dp_m_rho_c;
  dF1 = dF1 + lam_av.z * dp_m_rho_c * (u_av - c_av*r1);
  dF2 = dF2 + lam_av.z * dp_m_rho_c * (v_av - c_av*r2);
  dF3 = dF3 + lam_av.z * dp_m_rho_c * (h_av - c_av*V);

  double half_face = 0.5*mag;

  flx[0] = half_face*( (rho_l*V_l             ) + (rho_r*V_r             )  - dF0);
  flx[1] = half_face*( (rho_l*u_l*V_l + p_l*r1) + (rho_r*u_r*V_r + p_r*r1)  - dF1);
  flx[2] = half_face*( (rho_l*v_l*V_l + p_l*r2) + (rho_r*v_r*V_r + p_r*r2)  - dF2);
  flx[3] = half_face*( (e_l+p_l)*V_l + (e_r+p_r)*V_r                        - dF3);

  // if(dir==1 and j==DBGJ and k==DBGK){
  //   printf("___roe___%d %d -- %24.16e %24.16e\n", j, k, V_l, V_r);
  // }

}

template<int dir>
__global__ void muscl(int jtot, int ktot, double* q, double* ql, double* qr){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int v  = threadIdx.z;

  if(j>jtot-1 or k>ktot-1) return;

  q   += blockIdx.z*jtot*ktot*4; // this is primitive q
  ql  += blockIdx.z*jtot*ktot*4;
  qr  += blockIdx.z*jtot*ktot*4;

  double eps   = 0.01;
  double third = 1.0/3.0;
  int grid_idx = j + k*jtot;
  int idx, idx_m1, idx_p1, stride, minb, maxb;
  double dq, dq1, num, denom, kor_lim;
  double thm = 1.0 - third;
  double thp = 1.0 + third;

  if(dir == 0){
    stride    = 4;
    minb      = (j==0);
    maxb      = (j==jtot-1);
  }
  if(dir == 1){
    stride    = jtot*4;
    minb      = (k==0);
    maxb      = (k==ktot-1);
  }

  idx    = grid_idx*4+v;
  idx_m1 = idx - stride;
  idx_p1 = idx + stride;

  if(minb){
    ql[idx_p1] = q[idx];
    idx_m1 = idx;
  } else if(maxb){
    qr[idx] = q[idx];
    idx_p1  = idx;
  } else {
    dq               = q[idx_p1] - q[idx]; 
    dq1              = q[idx]    - q[idx_m1];
    num              = 3.0*(dq*dq1 + eps);
    denom            = 2.0*(dq - dq1)*(dq - dq1) + num;
    kor_lim          = 0.25 * num / denom;
    ql[idx_p1] = q[idx] + kor_lim*( thm*dq1 + thp*dq );
    qr[idx]    = q[idx] - kor_lim*( thp*dq1 + thm*dq );
  }

  // if(dir==0 and j==DBGJ and k==DBGK and v==1){
  //   printf("\n");
  //   printf("___mcl___%d %d %d -- %24.16e %24.16e\n", j, k, idx, ql[idx_p1], qr[idx]);
  //   printf("\n");
  // }


}

//
// Device Function to do the WENO interpolation
//
__device__ double weno_interpolation(double a,double b,double c,double d,double e)
{
  double thirteen_by_twelve = 13.0/12.0;
  double one_sixth = 1.0/6.0; 
  double epsw = 0.000001;

  double tau;
#ifdef BORGES_WEIGHT
  epsw = 1.e-20;
#endif

#ifndef UPWIND_OPT
  double a_minus_2b_plus_c  = a - 2.0*b + c;
  double a_minus_4b_plus_3c = a - 4.0*b + 3.0*c; 
  double b_minus_2c_plus_d  = b - 2.0*c + d;
  double b_minus_d          = b - d;
  double c_minus_2d_plus_e  = c - 2.0*d + e;
  double c3_minus_4d_plus_e = 3.0*c - 4.0*d + e;

  double dis0 = ( thirteen_by_twelve*a_minus_2b_plus_c*a_minus_2b_plus_c + 
		  0.25*a_minus_4b_plus_3c*a_minus_4b_plus_3c + epsw );
  double dis1 = 1.0/(thirteen_by_twelve*b_minus_2c_plus_d*b_minus_2c_plus_d + 
		     0.25*b_minus_d*b_minus_d + epsw);
  double dis2 = 1.0/(thirteen_by_twelve*c_minus_2d_plus_e*c_minus_2d_plus_e + 
		     0.25*c3_minus_4d_plus_e*c3_minus_4d_plus_e + epsw);
#endif

#ifdef BORGES_WEIGHT
  tau = fabs(dis2-dis0);
  dis0 = 1.0/(1.0+pow((tau/dis0),2.0)); 
  dis1 = (1.0+pow((tau*dis1),2.0)); 
  dis2 = (1.0+pow((tau*dis2),2.0)); 
#endif 

#ifdef UPWIND_OPT
  double w0 = 1.0;
  double w1 = 1.0;
  double w2 = 1.0;
#else
  double w0 = 1.0/(1.0 + 6.0*dis0*dis1 + 3.0*dis0*dis2);
  double w1 = 6.0*dis0*dis1*w0;
  double w2 = 1.0 - w0 - w1;
#endif 

  double weno = one_sixth*(w0*(2.0*a - 7.0*b + 11.0*c) + 
			   w1*(-b + 5.0*c + 2.0*d)     + 
			   w2*(2.0*c + 5.0*d - e)     );
  return weno;
}


template<int dir>
__global__ void weno5(int jtot, int ktot, double* q, double* ql, double* qr){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int v  = threadIdx.z;

  if(j>jtot-1 or k>ktot-1) return;

  q   += blockIdx.z*jtot*ktot*4; // this is primitive q
  ql  += blockIdx.z*jtot*ktot*4;
  qr  += blockIdx.z*jtot*ktot*4;

  int idx, idx_m1, idx_p1, idx_m2, idx_p2, idx_m3;

  idx = (j + k*jtot)*4+v;

  if(dir == 0){
    idx_m1 = idx    - 4*(j>0);
    idx_m2 = idx_m1 - 4*(j>1);
    idx_m3 = idx_m2 - 4*(j>2);
    idx_p1 = idx    + 4*(j<jtot-2);
    idx_p2 = idx_p1 + 4*(j<jtot-3);
  }
  if(dir == 1){
    idx_m1 = idx    - jtot*4*(k>0);
    idx_m2 = idx_m1 - jtot*4*(k>1);
    idx_m3 = idx_m2 - jtot*4*(k>2);
    idx_p1 = idx    + jtot*4*(k<ktot-2);
    idx_p2 = idx_p1 + jtot*4*(k<ktot-3);
  }

  ql[idx] = weno_interpolation(q[idx_m3],q[idx_m2],q[idx_m1],q[idx],q[idx_p1]);
  qr[idx] = weno_interpolation(q[idx_p2],q[idx_p1],q[idx],q[idx_m1],q[idx_m2]);

}


__global__ void to_prim(int jtot, int ktot, int nvar, double* q, double* qprim){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  if(j>jtot-1 or k>ktot-1) return;

  q      += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  qprim  += j*4    + k*jtot*4    + blockIdx.z*jtot*ktot*4;

  // int idx = j*4    + k*jtot*4    + blockIdx.z*jtot*ktot*4;

  qprim[0] = q[0];
  qprim[1] = q[1]/q[0];
  qprim[2] = q[2]/q[0];
  qprim[3] = (GAMMA - 1.0)*(q[3] - 0.5*(q[1]*q[1] + q[2]*q[2])/q[0]);

}

template<int dir>
__global__ void add_iflux(int jtot, int ktot, int nvar, int nghost, double* s, double* flx, double* vol){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;
  int v  = threadIdx.z;

  if(j+nghost > jtot-1 or k+nghost > ktot-1) return;

  s   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  flx += j*4    + k*jtot*4    + blockIdx.z*jtot*ktot*4;

  int stride = (dir==0)? 1 : jtot;

  s[v] -= (flx[stride*4+v] - flx[v])/vol[j+k*jtot];
  // s[v] -= (flx[stride*4+v] - flx[v]);

}

void G2D::inviscid_flux(double* q, double* s){

  int primcount = nl*jtot*ktot*4;

  int c=0;
  double* ql    = &wrk[c]; c+= primcount;
  double* qr    = &wrk[c]; c+= primcount;
  double* qprim = &wrk[c]; c+= primcount;
  double* flx   = &wrk[c]; c+= primcount;

  dim3 vthr(32,4,4);
  dim3 thr(32,16,1);
  dim3 vblk, blk, vblk_ng;
  blk.x     = (jtot-1)/thr.x+1; // initially we want all points (even ghosts)
  blk.y     = (ktot-1)/thr.y+1; //
  blk.z     = nl;
  vblk.x    = (jtot-1)/vthr.x+1;
  vblk.y    = (ktot-1)/vthr.y+1;
  vblk.z    = nl;
  vblk_ng.x = (jtot-1-nghost*2)/vthr.x+1;
  vblk_ng.y = (ktot-1-nghost*2)/vthr.y+1;
  vblk_ng.z = nl;

  // Compute primitives
  to_prim<<<blk,thr>>>(jtot,ktot,nvar,q,qprim);

  blk.x     = (jtot-nghost*2)/thr.x+1; // for roe flux kernel we want no ghost but we do 
  blk.y     = (ktot-nghost*2)/thr.y+1; // want the last face so this is not jtot-1-nghost*2
  blk.z     = nl;

  // 
  // J-Direction
  //
  if(order<5) muscl<0><<<vblk,vthr>>>(jtot,ktot,qprim,ql,qr);
  else        weno5<0><<<vblk,vthr>>>(jtot,ktot,qprim,ql,qr);
  roe_flux<0><<<blk,thr>>>(jtot,ktot,nvar,nghost,ql,qr,flx,Sj);
  add_iflux<0><<<vblk_ng,vthr>>>(jtot,ktot,nvar,nghost,s,flx,vol);

  // 
  // K-Direction
  //
  if(order<5) muscl<1><<<vblk,vthr>>>(jtot,ktot,qprim,ql,qr);
  else        weno5<1><<<vblk,vthr>>>(jtot,ktot,qprim,ql,qr);
  roe_flux<1><<<blk,thr>>>(jtot,ktot,nvar,nghost,ql,qr,flx,Sk);
  add_iflux<1><<<vblk_ng,vthr>>>(jtot,ktot,nvar,nghost,s,flx,vol);

}
