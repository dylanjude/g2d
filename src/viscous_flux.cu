#include "g2d.h"

#define DBGJ 97
#define DBGK 2

__device__ void fill_shared_vf(int jtot, int ktot, int nvar, double* qs, double* q, int pad){

  int tot_threads = blockDim.x*blockDim.y*blockDim.z; 

  // These are the dimensions we want to fill:
  int mem_jtot = blockDim.x+pad*2;
  int mem_ktot = blockDim.y+pad*2;
  int tot_points = mem_jtot*mem_ktot;
  
  // Linear Index from kernel blocks
  int mem_idx = threadIdx.x + threadIdx.y*blockDim.x;

  int jj, kk, j, k, qidx, msol_idx;

  double rho,u,v,p;

  while(mem_idx < tot_points){

    // Index in Shared Memory
    jj = mem_idx%mem_jtot;
    kk = ((mem_idx-jj)/mem_jtot)%mem_ktot;

    msol_idx = mem_idx*4;

    // Indices in real memory
    j = blockDim.x * blockIdx.x + jj - pad;
    k = blockDim.y * blockIdx.y + kk - pad;

    qidx = (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;

    // Check we're within the bounds
    if(j >= 0 && k >= 0 && j < jtot && k < ktot) {

      rho = q[qidx  ];
      u   = q[qidx+1]/rho;
      v   = q[qidx+2]/rho;
      p   = (GAMMA - 1.0)*(q[qidx+3] - 0.5*rho*(u*u+v*v));

      qs[msol_idx  ] = rho;
      qs[msol_idx+1] = u;
      qs[msol_idx+2] = v;
      qs[msol_idx+3] = GAMMA*p/rho; // this is like temperature

    }
    // now increment by the total threads
    mem_idx += tot_threads;
  }
} // done filling shared


__global__ void compute_viscous_flux(int jtot, int ktot, int nvar, int nghost, double* q, double* rhs, 
				     double2* Sj, double2* Sk, double* vol, double* flx_x, double* flx_y, 
				     double* mulam, double* muturb,
				     double* machs, double* reys, int nM, int nAoa){

  extern __shared__ double qs[];

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  int im = blockIdx.z%nM;
  // int ia = (blockIdx.z/nM)%nAoa;
  int ir = blockIdx.z/(nM*nAoa);

  double rey = reys[ir]/machs[im]; // reynolds number based on Mach

  int in_range = (j >= nghost and j+nghost <= jtot and k >= nghost and k+nghost <= ktot);

  int jsstride = 4;
  int ksstride = 4*(blockDim.x+2);

  fill_shared_vf(jtot,ktot,nvar,qs,q,1);

  __syncthreads(); // don't forget!

  if(!in_range) return;

  flx_x += (j + k*jtot + blockIdx.z*jtot*ktot)*3;
  flx_y += (j + k*jtot + blockIdx.z*jtot*ktot)*3;

  mulam  += blockIdx.z*jtot*ktot;
  muturb += blockIdx.z*jtot*ktot;

  int j_smem = threadIdx.x + 1;
  int k_smem = threadIdx.y + 1;

  int idxs_p0_p0 = j_smem*jsstride + k_smem*ksstride;
  int idxs_m1_p0 = idxs_p0_p0 - jsstride           ;
  int idxs_p1_p0 = idxs_p0_p0 + jsstride           ;
  int idxs_p0_m1 = idxs_p0_p0            - ksstride;
  int idxs_p0_p1 = idxs_p0_p0            + ksstride;
  int idxs_m1_p1 = idxs_p0_p0 - jsstride + ksstride;
  int idxs_m1_m1 = idxs_p0_p0 - jsstride - ksstride;
  int idxs_p1_m1 = idxs_p0_p0 + jsstride - ksstride;

  double inv_Reynolds     = 1.0/rey;
  double inv_Prandtl      = 1.0/PRANDTL;
  double inv_turb_Prandtl = 1.0/TURB_PRANDTL;
  double fac              = inv_Reynolds*2.0/3.0;

  double u, v;
  double u_xi, v_xi, T_xi;
  double u_eta, v_eta, T_eta;
  double nx, ny;

  double xi_x, xi_y;
  double eta_x, eta_y;

  double dudx, dudy;
  double dvdx, dvdy;

  double lam_visc, turb_visc, total_visc, ndim_stress;
  double tau_xx, tau_xy, tau_yy;
  double q_x, q_y;

  double total_Prandtl_visc, ndim_temp_grad;
  double jac, midjac, qmidjac;

  jac = 1.0/vol[j + k*jtot];

  int grid_idx_m1;
  int grid_idx = j + k*jtot;
  int jstride  = 1;
  int kstride  = jtot;

  // *************************************************************
  // Xi Direction Fluxes
  //
  grid_idx_m1 = grid_idx - jstride;
  midjac      = 0.5*(jac + 1.0/vol[grid_idx_m1]);
  qmidjac     = 0.25*midjac;

  nx = -Sj[j + k*jtot].x;
  ny = -Sj[j + k*jtot].y;

  u = 0.5*(qs[idxs_p0_p0 + 1] + qs[idxs_m1_p0 + 1]);
  v = 0.5*(qs[idxs_p0_p0 + 2] + qs[idxs_m1_p0 + 2]);
  //
  u_xi = qs[idxs_p0_p0 + 1] - qs[idxs_m1_p0 + 1];
  v_xi = qs[idxs_p0_p0 + 2] - qs[idxs_m1_p0 + 2]; 
  T_xi = qs[idxs_p0_p0 + 3] - qs[idxs_m1_p0 + 3]; 

  u_eta = 0.25*(qs[idxs_m1_p1+1]+qs[idxs_p0_p1+1]-qs[idxs_m1_m1+1]-qs[idxs_p0_m1+1]);
  v_eta = 0.25*(qs[idxs_m1_p1+2]+qs[idxs_p0_p1+2]-qs[idxs_m1_m1+2]-qs[idxs_p0_m1+2]);
  T_eta = 0.25*(qs[idxs_m1_p1+3]+qs[idxs_p0_p1+3]-qs[idxs_m1_m1+3]-qs[idxs_p0_m1+3]);

  //
  xi_x = -nx*midjac;
  xi_y = -ny*midjac;

  eta_x  = qmidjac*( Sk[grid_idx].x         + Sk[grid_idx_m1].x + 
  		     Sk[grid_idx+kstride].x + Sk[grid_idx_m1+kstride].x);
  eta_y  = qmidjac*( Sk[grid_idx].y         + Sk[grid_idx_m1].y + 
  		     Sk[grid_idx+kstride].y + Sk[grid_idx_m1+kstride].y);
  
  dudx = xi_x*u_xi + eta_x*u_eta;
  dudy = xi_y*u_xi + eta_y*u_eta;

  dvdx = xi_x*v_xi + eta_x*v_eta;
  dvdy = xi_y*v_xi + eta_y*v_eta;

  //
  lam_visc    = 0.5*(mulam[grid_idx] + mulam[grid_idx_m1]);
  turb_visc   = 0.5*(muturb[grid_idx] + muturb[grid_idx_m1]);
  total_visc  = lam_visc + turb_visc;
  ndim_stress = fac*total_visc;

  tau_xx = ndim_stress*(2.0*dudx - dvdy); 
  tau_yy = ndim_stress*(2.0*dvdy - dudx);
  tau_xy = 1.5*ndim_stress*(dudy + dvdx); 

  total_Prandtl_visc = inv_Prandtl*lam_visc + inv_turb_Prandtl*turb_visc;
  ndim_temp_grad     = -inv_Reynolds*total_Prandtl_visc/(GAMMA-1.0);

  q_x = ndim_temp_grad*(xi_x*T_xi + eta_x*T_eta);
  q_y = ndim_temp_grad*(xi_y*T_xi + eta_y*T_eta);

  flx_x[0] = (-(nx*tau_xx + ny*tau_xy));
  flx_x[1] = (-(nx*tau_xy + ny*tau_yy));
  flx_x[2] = (-nx*(u*tau_xx + v*tau_xy - q_x)
	      -ny*(u*tau_xy + v*tau_yy - q_y));	      

  // if(j==DBGJ and k==DBGK){
  //   printf("__der__ %18.8e %18.8e\n", dudx, dvdy);
  // }

  // *************************************************************
  // Eta Direction Fluxes
  //
  grid_idx_m1 = grid_idx - kstride;
  midjac      = 0.5*(jac + 1.0/vol[grid_idx_m1]);
  qmidjac     = 0.25*midjac;

  nx = -Sk[grid_idx].x;
  ny = -Sk[grid_idx].y;

  u = 0.5*(qs[idxs_p0_p0 + 1] + qs[idxs_p0_m1 + 1]);
  v = 0.5*(qs[idxs_p0_p0 + 2] + qs[idxs_p0_m1 + 2]);
  //
  u_xi = 0.25*(qs[idxs_p1_m1+1]+qs[idxs_p1_p0+1]-qs[idxs_m1_m1+1]-qs[idxs_m1_p0+1]);
  v_xi = 0.25*(qs[idxs_p1_m1+2]+qs[idxs_p1_p0+2]-qs[idxs_m1_m1+2]-qs[idxs_m1_p0+2]);
  T_xi = 0.25*(qs[idxs_p1_m1+3]+qs[idxs_p1_p0+3]-qs[idxs_m1_m1+3]-qs[idxs_m1_p0+3]);

  u_eta = qs[idxs_p0_p0+1]-qs[idxs_p0_m1+1];
  v_eta = qs[idxs_p0_p0+2]-qs[idxs_p0_m1+2];
  T_eta = qs[idxs_p0_p0+3]-qs[idxs_p0_m1+3];

  
  xi_x   = qmidjac*( Sj[grid_idx].x         + Sj[grid_idx_m1].x + 
  		     Sj[grid_idx+jstride].x + Sj[grid_idx_m1+jstride].x);
  xi_y   = qmidjac*( Sj[grid_idx].y         + Sj[grid_idx_m1].y + 
  		     Sj[grid_idx+jstride].y + Sj[grid_idx_m1+jstride].y);

  eta_x = -nx*midjac;
  eta_y = -ny*midjac;

  dudx = xi_x*u_xi + eta_x*u_eta;
  dudy = xi_y*u_xi + eta_y*u_eta;

  dvdx = xi_x*v_xi + eta_x*v_eta;
  dvdy = xi_y*v_xi + eta_y*v_eta;

  //
  lam_visc    = 0.5*(mulam[grid_idx] + mulam[grid_idx_m1]);
  turb_visc   = 0.5*(muturb[grid_idx] + muturb[grid_idx_m1]);
  total_visc  = lam_visc + turb_visc;
  ndim_stress = fac*total_visc;

  tau_xx = ndim_stress*(2.0*dudx - dvdy); 
  tau_yy = ndim_stress*(2.0*dvdy - dudx);
  tau_xy = 1.5*ndim_stress*(dudy + dvdx); 

  // if(j==DBGJ and k==DBGK){
  //   printf("__vfl__ %24.16e %24.16e %24.16e \n", lam_visc, mulam[grid_idx], mulam[grid_idx_m1]);
  // }

  total_Prandtl_visc = inv_Prandtl*lam_visc + inv_turb_Prandtl*turb_visc;
  ndim_temp_grad     = -inv_Reynolds*total_Prandtl_visc/(GAMMA-1.0);

  q_x = ndim_temp_grad*(xi_x*T_xi + eta_x*T_eta);
  q_y = ndim_temp_grad*(xi_y*T_xi + eta_y*T_eta);

  flx_y[0] = (-(nx*tau_xx + ny*tau_xy));
  flx_y[1] = (-(nx*tau_xy + ny*tau_yy));
  flx_y[2] = (-nx*(u*tau_xx + v*tau_xy - q_x)
	      -ny*(u*tau_xy + v*tau_yy - q_y));

}

#define SUTH_C2B 0.3678
__global__ void compute_mulam(int jtot, int ktot, int nvar, double* q, double* mulam){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(j > jtot-1 or k > ktot-1) return;

  q     += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;
  mulam +=  j + k*jtot + blockIdx.z*jtot*ktot;

  double T = (GAMMA)*(GAMMA-1.0)*( q[3] - 0.5*(q[1]*q[1]+q[2]*q[2])/q[0] )/q[0];
  mulam[0] = (SUTH_C2B+1.0)*T*sqrt(T)/(SUTH_C2B+T);

}

__global__ void add_vflux(int jtot, int ktot, int nvar, int nghost, double* s, double* flx_x, double* flx_y, double* vol){

  int j  = blockDim.x*blockIdx.x + threadIdx.x + nghost;
  int k  = blockDim.y*blockIdx.y + threadIdx.y + nghost;
  int v  = threadIdx.z;

  if(j+nghost > jtot-1 or k+nghost > ktot-1) return;

  s      += (j      + k*jtot      + blockIdx.z*jtot*ktot)*nvar;
  flx_x  += (j      + k*jtot      + blockIdx.z*jtot*ktot)*3;
  flx_y  += (j      + k*jtot      + blockIdx.z*jtot*ktot)*3;

  s[v+1] += (flx_x[v+3     ] - flx_x[v])/vol[j+k*jtot];
  s[v+1] += (flx_y[v+3*jtot] - flx_y[v])/vol[j+k*jtot];
  // s[v+1] += (flx_x[v+3     ] - flx_x[v]);
  // s[v+1] += (flx_y[v+3*jtot] - flx_y[v]);

}

void G2D::set_mulam(double* q){
  int nl        = nM*nRey*nAoa;
  dim3 thr(16,16,1);
  dim3 blk(1,1,nl);
  blk.x     = (jtot-1)/thr.x+1; 
  blk.y     = (ktot-1)/thr.y+1; 
  compute_mulam<<<blk,thr>>>(jtot,ktot,nvar,q,mulam);
}

void G2D::viscous_flux(double* q, double* s){

  int nl        = nM*nRey*nAoa;
  int count     = nl*jtot*ktot;

  int c=0;
  double* flx_x = &wrk[c]; c+= count*3;
  double* flx_y = &wrk[c]; c+= count*3;

  dim3 thr(16,16,1);
  dim3 blk(1,1,nl);
  blk.x     = (jtot-1)/thr.x+1; // initially we want all points (even ghosts)
  blk.y     = (ktot-1)/thr.y+1; //

  size_t mem = (thr.x+2) * (thr.y+2) * 4 * sizeof(double);

  // mulam and muturb should previously already been calculated

  compute_viscous_flux<<<blk,thr,mem>>>(jtot,ktot,nvar,nghost,q,s,Sj,Sk,vol,flx_x,flx_y,mulam,muturb,machs[GPU],reys[GPU],nM,nAoa);

  // debug_print(87,2,0,flx_x,3);
  // debug_print(87,2,0,flx_y,3);

  dim3 vthr(32,4,3);
  dim3 vblk_ng;
  vblk_ng.x = (jtot-1-nghost*2)/vthr.x+1;
  vblk_ng.y = (ktot-1-nghost*2)/vthr.y+1;
  vblk_ng.z = nl;

  // debug_print(97,2,0,s,5);

  add_vflux<<<vblk_ng,vthr>>>(jtot,ktot,nvar,nghost,s,flx_x,flx_y,vol);

  // debug_print(97,2,0,s,5);
  // debug_print(97,2,0,flx_y,3);

}
