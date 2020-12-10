#include "g2d.h"

#define DBGJ 87
#define DBGK 42

// #define DADI_REDUCED_ORDER
#define EPSLAM 0.08
#define NFDADI 1

#ifdef DADI_REDUCED_ORDER
#define dadireal float
#else
#define dadireal double
#endif

__global__ void times_dt(int jtot,int ktot,int nvar,int nghost,double* s, double* dt){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;
  int v  = threadIdx.z;

  s   += j*nvar + k*jtot*nvar + blockIdx.z*jtot*ktot*nvar;
  dt  += j      + k*jtot      + blockIdx.z*jtot*ktot;

  if(j+nghost < jtot and k+nghost < ktot){
    // for(int v=0; v<nvar; v++){
    s[v] *= dt[0];
    // }
  }
}

// Computes T matrix
// T_k in "A Diagonal Form of an Implicit Approximate-Factorization Algorithm" (Eqn. 9c)
__device__ void dadi_T(dadireal* T, dadireal rho, dadireal u, dadireal v, dadireal c, dadireal gam, dadireal kx, dadireal ky){

  dadireal gamma_minus_one     = gam - 1.0;
  dadireal phi_squared         = 0.5 * gamma_minus_one * (u*u + v*v);
  dadireal theta               = kx*u + ky*v;
  dadireal alpha               = rho/(sqrt(2.0)*c);
  dadireal inv_gamma_minus_one = 1.0/gamma_minus_one;

  //T
  //| 0  1  2  3 |
  //| 4  5  6  7 |
  //| 8  9 10 11 |
  //|12 13 14 15 |

  T[ 0] = 1.0;
  T[ 1] = 0.0;
  T[ 2] = alpha;
  T[ 3] = alpha;

  T[ 4] = u;
  T[ 5] = ky*rho;
  T[ 6] = alpha*(u + kx*c);
  T[ 7] = alpha*(u - kx*c);

  T[ 8] = v;
  T[ 9] = -kx*rho;
  T[10] = alpha*(v + ky*c);
  T[11] = alpha*(v - ky*c);

  T[12] = phi_squared*inv_gamma_minus_one;
  T[13] = rho*(ky*u - kx*v);
  T[14] = alpha*( (phi_squared + c*c)*inv_gamma_minus_one + c*theta);
  T[15] = alpha*( (phi_squared + c*c)*inv_gamma_minus_one - c*theta);

}

__device__ void Tk_inv_Tl(dadireal kx, dadireal ky, dadireal lx, dadireal ly, dadireal* N){

  dadireal m1, m2, u;
  m1 = kx*lx + ky*ly;
  m2 = kx*ly - ky*lx;
  u = 1.0/sqrt(2.0);     // this is mu

  N[ 0] =  1.0;
  N[ 1] =  0.0;
  N[ 2] =  0.0;
  N[ 3] =  0.0;

  N[ 4] =  0.0;
  N[ 5] =  m1;
  N[ 6] = -u*m2;
  N[ 7] =  u*m2;

  N[ 8] =  0.0;
  N[ 9] =  u*m2;
  N[10] =  u*u*(1.0+m1);
  N[11] =  u*u*(1.0-m1);

  N[12] =  0.0;
  N[13] = -u*m2;
  N[14] =  u*u*(1.0-m1);
  N[15] =  u*u*(1.0+m1);

}


// Computes T_inv matrix
// T_k_inv in "A Diagonal Form of an Implicit Approximate-Factorization Algorithm" (Eqn. 9d)
__device__ void dadi_T_inv(dadireal* T_inv, dadireal rho, dadireal u, dadireal v, 
			      dadireal c, dadireal gam, dadireal k_x, dadireal k_y){

  dadireal gamma_minus_one = gam - 1.0;
  dadireal phi_squared     = 0.5 * gamma_minus_one * (u*u + v*v);
  dadireal theta           = k_x*u + k_y*v;

  dadireal beta = 1.0/(sqrt(2.0)*rho*c);
  dadireal inv_rho = 1.0/rho;
  dadireal inv_c_squared = 1.0/(c*c);

  //T_inv
  //| 0  1  2  3 |
  //| 4  5  6  7 |
  //| 8  9 10 11 |
  //|12 13 14 15 |

  T_inv[ 0] =  (1.0 - phi_squared*inv_c_squared);
  T_inv[ 1] =  gamma_minus_one*u*inv_c_squared;
  T_inv[ 2] =  gamma_minus_one*v*inv_c_squared;
  T_inv[ 3] = -gamma_minus_one*inv_c_squared; 

  T_inv[ 4] = -inv_rho*(k_y*u - k_x*v);
  T_inv[ 5] =  k_y*inv_rho;
  T_inv[ 6] = -k_x*inv_rho;
  T_inv[ 7] =  0;

  T_inv[ 8] =  beta*(phi_squared - c*theta);
  T_inv[ 9] =  beta*(k_x*c - gamma_minus_one*u);
  T_inv[10] =  beta*(k_y*c - gamma_minus_one*v);
  T_inv[11] =  beta*gamma_minus_one;

  T_inv[12] =  beta*(phi_squared + c*theta);
  T_inv[13] = -beta*(k_x*c + gamma_minus_one*u); 
  T_inv[14] = -beta*(k_y*c + gamma_minus_one*v); 
  T_inv[15] =  beta*gamma_minus_one; 

}

// Solve a scalar tridiagonal system of equations using the Thomas algorithm
// A*S = R -> S = A_inv*R
//
//     |D U      |
//     |L D U    |
// A = |  L D U  |
//     |    L D U|
//     |      L D|
__device__ void solve_tridiagonal(dadireal* L, dadireal* D, dadireal* U, dadireal* R, 
				  int imax, int istride, int idx_start){
  // Initialize for forward sweep
  U[idx_start] = U[idx_start]/D[idx_start];
  R[idx_start] = R[idx_start]/D[idx_start];

  // Forward sweep
  for (int i = 1; i <= imax - 1; i++)
  {
    int idx    = idx_start + i*istride;
    int idx_m1 = idx - istride;

    dadireal inv_main_diag = 1.0/(D[idx] - L[idx]*U[idx_m1]);

    U[idx] = U[idx]*inv_main_diag;
    R[idx] = (R[idx] - L[idx]*R[idx_m1])*inv_main_diag;
  }

  // Backward sweep
  for (int i = imax - 2; i >= 0; i--)
  {
    int idx    = idx_start + i*istride;
    int idx_p1 = idx + istride;

    R[idx] = R[idx] - U[idx]*R[idx_p1];
  }
}

// Periodic tri-diagonal solver: Shermann-Morrison Alg
__device__ void solve_ptridiagonal(dadireal *a, dadireal* b, dadireal* c, dadireal* f, 
				   int N, int stride, int ks){

  int k, km, kp, ke, kem;
  dadireal bb;
  ke  = ks+(N-1)*stride;
  kem = ke-stride;
  
  bb    = 1.0/b[ks];
  c[ks] = c[ks]*bb;
  f[ks] = f[ks]*bb;
  a[ks] = a[ks]*bb;
  
  for(k=ks+stride; k<=kem; k+=stride){
    km = k-stride;
    bb = 1.0/(b[k]-a[k]*c[km]);
    c[k] = c[k]*bb;
    f[k] = (f[k]- a[k]*f[km])*bb;
    a[k] = -a[k]*a[km]*bb;
  }

  a[kem] = a[kem] + c[kem];

  for(k=ke-2*stride; k>=ks; k-=stride){
    kp = k+stride;
    a[k] = a[k]-c[k]*a[kp];
    f[k] = f[k]-c[k]*f[kp];
  }

  bb    = ( (f[ke]-a[ke]*f[kem]-c[ke]*f[ks]) / 
	    (b[ke]-a[ke]*a[kem]-c[ke]*a[ks]) );
  f[ke] = bb;
  // f[ke] = ( (f[ke]-a[ke]*f[kem]-c[ke]*f[ks]) / 
  // 	    (b[ke]-a[ke]*a[kem]-c[ke]*a[ks]) );

  for(k=ks; k<=kem; k+=stride){
    f[k] = f[k]-a[k]*bb;
  }

}

template<int dir>
__global__ void compute_LDU(int jtot, int ktot, int nvar, double2* S, dadireal* L_DADI, dadireal* D_DADI, dadireal* U_DADI, 
			    double* vol, double* dt_array, double* q, bool visc){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(j > jtot-1-2*NFDADI or k > ktot-1-2*NFDADI) return;

  j += NFDADI;
  k += NFDADI;

  dt_array += blockIdx.z*jtot*ktot; // so we can index with grid_idx later

  q += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;

  int grid_idx = j + k*jtot;
  int soln_idx = (j + k*jtot + blockIdx.z*jtot*ktot)*4;
  int stride   = (dir==0)? 1 : jtot;

  int grid_idx_p1 = grid_idx + stride;
  int grid_idx_m1 = grid_idx - stride;
  int soln_idx_p1 = grid_idx_p1 * 4;
  int soln_idx_m1 = grid_idx_m1 * 4;

  dadireal jac    = 1.0/vol[grid_idx];

  dadireal k_x = 0.5*(S[grid_idx].x + S[grid_idx+stride].x)*jac;
  dadireal k_y = 0.5*(S[grid_idx].y + S[grid_idx+stride].y)*jac;
  dadireal mag = sqrt(k_x*k_x + k_y*k_y);

  dadireal rho     = q[0];
  dadireal u       = q[1]/rho;
  dadireal v       = q[2]/rho;
  dadireal p       = (GAMMA - 1.0)*(q[3] - 0.5*rho*(u*u+v*v));

  dadireal cn = sqrt(GAMMA * p / rho)*mag;

  dadireal vis_L = 0.0;
  dadireal vis_D = 0.0;
  dadireal vis_U = 0.0;

  // if (visc){
  //   dadireal inv_rho = 1.0/q[soln_idx];
  //   dadireal rho_m1  = q[soln_idx_m1];
  //   dadireal rho_p1  = q[soln_idx_p1];
  //   dadireal over_RePr  = 1.0 / ( Reynolds * Prandtl);
  //   dadireal Sp, Sm, S0; // face sizes squared
  //   Sm = dot(S[grid_idx_m1], S[grid_idx_m1])*d_grid->jacobian[grid_idx_m1];
  //   S0 = dot(S[grid_idx   ], S[grid_idx   ])*d_grid->jacobian[grid_idx];
  //   Sp = dot(S[grid_idx_p1], S[grid_idx_p1])*d_grid->jacobian[grid_idx_p1];
  //   dadireal vnu    = ( mu_laminar[grid_idx] + 
  // 		      mu_laminar[grid_idx_p1] + 
  //   		      nu_turbulent[grid_idx]*rho  +
  // 		      nu_turbulent[grid_idx_p1]*rho_p1 ) * over_RePr * 0.5;
  //   dadireal vnu_m1 = ( mu_laminar[grid_idx] + 
  // 		      mu_laminar[grid_idx_m1] + 
  //   		      nu_turbulent[grid_idx]*rho  +
  // 		      nu_turbulent[grid_idx_m1]*rho_m1 ) * over_RePr * 0.5;
  //   dadireal d1 = 0.5 * (S0 + Sp) * GAMMA * vnu    * inv_rho * jac;
  //   dadireal d2 = 0.5 * (S0 + Sm) * GAMMA * vnu_m1 * inv_rho * jac;
  //   vis_L  = - d1;
  //   vis_U  = - d2;
  //   vis_D  = d1 + d2;
  // }


  // From Eqn 17b in "A Diagonal Form of an Implicit Approximate-Factorization Algorithm"
  dadireal lam0, lam2, lam3;
  lam0 = k_x*u + k_y*v;
  lam2 = lam0 + cn;
  lam3 = lam0 - cn;

  dadireal eps = EPSLAM * mag; // <-- because turns does it
  dadireal abs_lambda, lambda_plus, lambda_minus;
  dadireal dt = dt_array[grid_idx];

  // lambda 0
  abs_lambda   = (1.0+EPSLAM)*sqrt(lam0*lam0 + eps*eps);
  lambda_plus  = 0.5*(lam0 + abs_lambda);
  lambda_minus = 0.5*(lam0 - abs_lambda);

  L_DADI[soln_idx  ] =       (vis_L - lambda_plus               )*dt;
  D_DADI[soln_idx  ] = 1.0 + (vis_D + lambda_plus - lambda_minus)*dt;
  U_DADI[soln_idx  ] =       (vis_U + lambda_minus              )*dt;
  // lambda 1 = lambda 0
  L_DADI[soln_idx+1] =       (vis_L - lambda_plus               )*dt;
  D_DADI[soln_idx+1] = 1.0 + (vis_D + lambda_plus - lambda_minus)*dt;
  U_DADI[soln_idx+1] =       (vis_U + lambda_minus              )*dt;

  // lambda 2
  abs_lambda   = (1.0+EPSLAM)*sqrt(lam2*lam2 + eps*eps);
  lambda_plus  = 0.5*(lam2 + abs_lambda);
  lambda_minus = 0.5*(lam2 - abs_lambda);

  L_DADI[soln_idx+2] =       (vis_L - lambda_plus               )*dt;
  D_DADI[soln_idx+2] = 1.0 + (vis_D + lambda_plus - lambda_minus)*dt;
  U_DADI[soln_idx+2] =       (vis_U + lambda_minus              )*dt;

  // lambda 3
  abs_lambda   = (1.0+EPSLAM)*sqrt(lam3*lam3 + eps*eps);
  lambda_plus  = 0.5*(lam3 + abs_lambda);
  lambda_minus = 0.5*(lam3 - abs_lambda);

  L_DADI[soln_idx+3] =       (vis_L - lambda_plus               )*dt;
  D_DADI[soln_idx+3] = 1.0 + (vis_D + lambda_plus - lambda_minus)*dt;
  U_DADI[soln_idx+3] =       (vis_U + lambda_minus              )*dt;

}

__global__ void invert_xi(int jtot, int ktot, int nvar, double2* Sj, double* q, dadireal* rhs){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(j > jtot-1-2*NFDADI or k > ktot-1-2*NFDADI) return;

  j += NFDADI;
  k += NFDADI;

  int grid_idx = j + k*jtot;

  q   += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;
  rhs += (j + k*jtot + blockIdx.z*jtot*ktot)*4;

  dadireal k_x = Sj[grid_idx].x;
  dadireal k_y = Sj[grid_idx].y;
  dadireal imag = 1.0/sqrt(k_x*k_x + k_y*k_y);
  k_x *= imag;
  k_y *= imag;
  dadireal T_inv[16];

  dadireal rho = q[0];
  dadireal u   = q[1]/rho;
  dadireal v   = q[2]/rho;
  dadireal p   = (GAMMA - 1.0)*(q[3] - 0.5*rho*(u*u+v*v));
  dadireal c   = sqrt(GAMMA * p / rho);

  dadi_T_inv(T_inv, rho, u, v, c, GAMMA, k_x, k_y);

  dadireal r0,r1,r2,r3;

  r0 = rhs[0];
  r1 = rhs[1];
  r2 = rhs[2];
  r3 = rhs[3];

  rhs[0] = T_inv[ 0]*r0 + T_inv[ 1]*r1 + T_inv[ 2]*r2 + T_inv[ 3]*r3;
  rhs[1] = T_inv[ 4]*r0 + T_inv[ 5]*r1 + T_inv[ 6]*r2 + T_inv[ 7]*r3;
  rhs[2] = T_inv[ 8]*r0 + T_inv[ 9]*r1 + T_inv[10]*r2 + T_inv[11]*r3;
  rhs[3] = T_inv[12]*r0 + T_inv[13]*r1 + T_inv[14]*r2 + T_inv[15]*r3;

}

__global__ void invert_eta(int jtot, int ktot, int nvar, double2* Sk, double* q, dadireal* rhs){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(j > jtot-1-2*NFDADI or k > ktot-1-2*NFDADI) return;

  j += NFDADI;
  k += NFDADI;

  int grid_idx = j + k*jtot;

  q   += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;
  rhs += (j + k*jtot + blockIdx.z*jtot*ktot)*4;

  dadireal k_x = Sk[grid_idx].x;
  dadireal k_y = Sk[grid_idx].y;
  dadireal imag = 1.0/sqrt(k_x*k_x + k_y*k_y);
  k_x *= imag;
  k_y *= imag;
  dadireal T[16];

  dadireal rho = q[0];
  dadireal u   = q[1]/rho;
  dadireal v   = q[2]/rho;
  dadireal p   = (GAMMA - 1.0)*(q[3] - 0.5*rho*(u*u+v*v));
  dadireal c   = sqrt(GAMMA * p / rho);

  dadi_T(T,rho, u, v, c, GAMMA, k_x, k_y);

  dadireal r0,r1,r2,r3;

  r0 = rhs[0];
  r1 = rhs[1];
  r2 = rhs[2];
  r3 = rhs[3];

  rhs[0] = T[ 0]*r0 + T[ 1]*r1 + T[ 2]*r2 + T[ 3]*r3;
  rhs[1] = T[ 4]*r0 + T[ 5]*r1 + T[ 6]*r2 + T[ 7]*r3;
  rhs[2] = T[ 8]*r0 + T[ 9]*r1 + T[10]*r2 + T[11]*r3;
  rhs[3] = T[12]*r0 + T[13]*r1 + T[14]*r2 + T[15]*r3;

}

__global__ void invert_xi_eta(int jtot, int ktot, double2* Sj, double2* Sk, dadireal* rhs){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  if(j > jtot-1-2*NFDADI or k > ktot-1-2*NFDADI) return;

  j += NFDADI;
  k += NFDADI;

  int grid_idx = j + k*jtot;

  rhs += (j + k*jtot + blockIdx.z*jtot*ktot)*4;

  dadireal imag;

  dadireal k_x = Sk[grid_idx].x;
  dadireal k_y = Sk[grid_idx].y;
  imag = 1.0/sqrt(k_x*k_x + k_y*k_y);
  k_x *= imag;
  k_y *= imag;

  dadireal l_x = Sj[grid_idx].x;
  dadireal l_y = Sj[grid_idx].y;
  imag = 1.0/sqrt(l_x*l_x + l_y*l_y);
  l_x *= imag;
  l_y *= imag;

  dadireal N[16];

  Tk_inv_Tl(k_x, k_y, l_x, l_y, N);

  dadireal r0,r1,r2,r3;

  r0 = rhs[0];
  r1 = rhs[1];
  r2 = rhs[2];
  r3 = rhs[3];
  rhs[0] = N[ 0]*r0 + N[ 1]*r1 + N[ 2]*r2 + N[ 3]*r3;
  rhs[1] = N[ 4]*r0 + N[ 5]*r1 + N[ 6]*r2 + N[ 7]*r3;
  rhs[2] = N[ 8]*r0 + N[ 9]*r1 + N[10]*r2 + N[11]*r3;
  rhs[3] = N[12]*r0 + N[13]*r1 + N[14]*r2 + N[15]*r3;
}


template<int dir, int per>
__global__ void tridiag(int jtot, int ktot, int nghost, dadireal *rhs, dadireal *L_DADI, dadireal *D_DADI, dadireal *U_DADI){

  int j, k;
  int var = threadIdx.x;
  int l   = blockIdx.z;
  if(dir == 0){
    k   = blockDim.y * blockIdx.y + threadIdx.y + NFDADI;
    if(k > ktot-1-NFDADI) return;
  }
  if(dir == 1){
    j   = blockDim.y * blockIdx.y + threadIdx.y + NFDADI;
    if(j > jtot-1-NFDADI) return;
  }

  int nf        = (per==0)? NFDADI      :  nghost;
  int n         = (dir==0)? jtot-2*nf   :  ktot-2*nf;
  int stride    = (dir==0)? 4           :  4*jtot;
  int idx_start = (dir==0)? nf + k*jtot :  j + nf*jtot;

  idx_start = (idx_start + l*jtot*ktot)*4 + var;

  if(per == 0){
    solve_tridiagonal(L_DADI,D_DADI,U_DADI,rhs,n,stride,idx_start);
  } 
  if(per == 1){
    solve_ptridiagonal(L_DADI,D_DADI,U_DADI,rhs,n,stride,idx_start);
  }

}

// if dadireal is double, all this does is re-order the array with nvar=4
template<bool todouble>
__global__ void change_precision(double *dbl, dadireal *flt, int nvar, int dof4){

  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if(idx < dof4){
     int qidx = (int(idx/4))*nvar + idx%4;
     if(todouble) dbl[qidx] = flt[idx ];
     else         flt[idx ] = dbl[qidx];
  }
}


void G2D::precondition(double* sin, double* sout){

  int nl     = nM*nRey*nAoa;
  int qcount = nl*jtot*ktot*nvar;
  int count4 = nl*jtot*ktot*4;

  if(sin != sout){
    HANDLE_ERROR( cudaMemcpy(sout, sin, qcount*sizeof(double), cudaMemcpyDeviceToDevice) );
  }

  dim3 vthr(32,4,nvar);
  dim3 vblk;
  vblk.x = (jtot-1-nghost*2)/vthr.x+1;
  vblk.y = (ktot-1-nghost*2)/vthr.y+1;
  vblk.z = nl;

  times_dt<<<vblk,vthr>>>(jtot,ktot,nvar,nghost,sout,dt);

  dim3 thr(32,16,1);
  dim3 blk;
  blk.x = (jtot-1-NFDADI*2)/thr.x+1;
  blk.y = (ktot-1-NFDADI*2)/thr.y+1;
  blk.z = nl;

  bool visc=false;

  //
  // DADI method
  //
  int c=0;
  dadireal *L_DADI = (dadireal*)(&wrk[c]); c+=count4;
  dadireal *D_DADI = (dadireal*)(&wrk[c]); c+=count4;
  dadireal *U_DADI = (dadireal*)(&wrk[c]); c+=count4;
  dadireal *R_DADI = (dadireal*)(&wrk[c]); c+=count4;
  int pts = jtot*ktot*nl*4;
  dim3 linblk(1,1,1), linthread(256,1,1);
  linblk.x = (pts-1)/linthread.x+1;
  change_precision<0><<<linblk,linthread>>>(sout, R_DADI, nvar, pts);

  dim3 triblocks(1,1,nl), trithreads(4,16,1);

  //
  // J-Direction: XI
  triblocks.y = (ktot-2*NFDADI)/trithreads.y+1; // Tri-diag solver kernel dims				
  invert_xi<<<blk,thr>>>(jtot,ktot,nvar,Sj,this->q[GPU],R_DADI);
  compute_LDU<0><<<blk,thr>>>(jtot,ktot,nvar,Sj,L_DADI,D_DADI,U_DADI,vol,dt,this->q[GPU],visc);
  tridiag<0,1><<<triblocks,trithreads>>>(jtot,ktot,nghost,R_DADI,L_DADI,D_DADI,U_DADI ); // note: periodic

  //
  // K-Direction: ETA
  triblocks.y = (jtot-2*NFDADI)/trithreads.y+1; // Tri-diag solver kernel dims				
  invert_xi_eta<<<blk,thr>>>(jtot,ktot,Sj,Sk,R_DADI);
  compute_LDU<1><<<blk,thr>>>(jtot,ktot,nvar,Sk,L_DADI,D_DADI,U_DADI,vol,dt,this->q[GPU],visc);
  tridiag<1,0><<<triblocks,trithreads>>>(jtot,ktot,nghost,R_DADI,L_DADI,D_DADI,U_DADI ); // note: periodic

  //
  // Last inversion 
  invert_eta<<<blk,thr>>>(jtot,ktot,nvar,Sk,this->q[GPU],R_DADI);

  change_precision<1><<<linblk,linthread>>>(sout, R_DADI, nvar, pts);

}
