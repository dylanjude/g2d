#include "g2d.h"

// #define USE_DES

#define ONE_BY_SIGMA 1.5
#define CB2 0.622
#define CB1 0.1355
#define CDES 0.65

#define DBGJ 110
#define DBGK 8

__device__ void fill_shared_sa(int jtot, int ktot, int nvar, double* q, double* mul, double* snul, double* snut, int pad){

  int tot_threads = blockDim.x*blockDim.y*blockDim.z; 

  // These are the dimensions we want to fill:
  int mem_jtot = blockDim.x+pad*2;
  int mem_ktot = blockDim.y+pad*2;
  int tot_points = mem_jtot*mem_ktot;
  
  // Linear Index from kernel blocks
  int mem_idx = threadIdx.x + threadIdx.y*blockDim.x;

  int jj, kk, j, k, qidx, grid_idx;

  while(mem_idx < tot_points){

    // Index in Shared Memory
    jj = mem_idx%mem_jtot;
    kk = ((mem_idx-jj)/mem_jtot)%mem_ktot;

    // Indices in real memory
    j = blockDim.x * blockIdx.x + jj - pad;
    k = blockDim.y * blockIdx.y + kk - pad;

    qidx     = (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;
    grid_idx = (j + k*jtot + blockIdx.z*jtot*ktot);

    // Check we're within the bounds
    if(j >= 0 && k >= 0 && j < jtot && k < ktot) {

      snul[mem_idx] = mul[grid_idx]/q[qidx];
      snut[mem_idx] = q[qidx+4];

    }
    // now increment by the total threads
    mem_idx += tot_threads;
  }
} // done filling shared


template<int mode>
__global__ void sa_conv_diff(int jtot, int ktot, int nvar, int nghost, double* q, double* rhs, double* LDU, double* mulam, 
			     double2* Sj, double2* Sk, double* vol,double* reys){

  extern __shared__ double smem[];

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  double *snul = &smem[0];
  double *snut = &smem[(blockDim.x+2)*(blockDim.y+2)];

  double rey = reys[blockIdx.z];

  int in_range = (j >= nghost and j+nghost < jtot and k >= nghost and k+nghost <= ktot);

  fill_shared_sa(jtot,ktot,nvar,q,mulam,snul,snut,1);  

  __syncthreads(); // don't forget!

  if(!in_range) return;

  q   += (blockIdx.z*jtot*ktot)*nvar;
  if(mode==0){
    rhs += (blockIdx.z*jtot*ktot)*nvar;
  } else {
    LDU += (blockIdx.z*jtot*ktot)*6;
  }

  int jstride = 1;
  int kstride = jtot;

  // strides in shared memory
  int jsstride = 1;
  int ksstride = (blockDim.x+2);

  // index of this cell in shared memory
  int sidx = (threadIdx.x+1)*jsstride + (threadIdx.y+1)*ksstride;

  //
  // SA Model constants
  //
  double cb2_by_sigma, one_plus_cb2_by_sigma;

  cb2_by_sigma           = ONE_BY_SIGMA/(rey);
  one_plus_cb2_by_sigma  = cb2_by_sigma*(1.0 + CB2);
  cb2_by_sigma           = cb2_by_sigma*(CB2);

  int grid_idx = j*jstride + k*kstride;
  int soln_idx = grid_idx*(nvar);

  int grid_idx_p1_j = grid_idx + jstride;
  int grid_idx_m1_j = grid_idx - jstride;
  int grid_idx_p1_k = grid_idx + kstride;
  int grid_idx_m1_k = grid_idx - kstride;

  double rho     = q[soln_idx];
  double uug     = q[soln_idx + 1]/rho;
  double vvg     = q[soln_idx + 2]/rho;
  double nu_lam  = snul[sidx];
  double nu_turb = snut[sidx];

  double midjac; 
  double jac = 1.0/vol[grid_idx];

  /// Metrics of the cell-center ///
  double   xi_x = 0.5*(Sj[grid_idx].x + Sj[grid_idx_p1_j].x)*jac;
  double   xi_y = 0.5*(Sj[grid_idx].y + Sj[grid_idx_p1_j].y)*jac;
  double  eta_x = 0.5*(Sk[grid_idx].x + Sk[grid_idx_p1_k].x)*jac;
  double  eta_y = 0.5*(Sk[grid_idx].y + Sk[grid_idx_p1_k].y)*jac;

  // *****************************************************************
  // Convection
  //
  double U =   xi_x*uug +   xi_y*vvg;
  double V =  eta_x*uug +  eta_y*vvg;

  double U_plus  = 0.5*(U + fabs(U));
  double U_minus = 0.5*(U - fabs(U));
  double V_plus  = 0.5*(V + fabs(V));
  double V_minus = 0.5*(V - fabs(V));

  double rhs_turb; 
  double D_turb_j, L_turb_j, U_turb_j, D_turb_k, L_turb_k, U_turb_k;

  if(mode==0){
    rhs_turb  = ( -U_plus*(snut[sidx]-snut[sidx-jsstride])-U_minus*(snut[sidx+jsstride]-snut[sidx])
		  -V_plus*(snut[sidx]-snut[sidx-ksstride])-V_minus*(snut[sidx+ksstride]-snut[sidx]));
  } else {
    // Augment J-direction Tridiagonal
    L_turb_j =  U_plus;
    D_turb_j = -U_plus + U_minus;
    U_turb_j = -U_minus;
    // Augment K-direction Tridiagonal
    L_turb_k =  V_plus;
    D_turb_k = -V_plus + V_minus;
    U_turb_k = -V_minus;                 
  }


  // *****************************************************************
  // Diffusion
  //
  double k_x_p_half, k_y_p_half;
  double k_x_m_half, k_y_m_half;

  double nu_laminar_p_half, nu_turbulent_tilda_p_half, chp_p_half;
  double nu_laminar_m_half, nu_turbulent_tilda_m_half, chp_m_half;
  double dnuhp, dnuhp_m1;

  double dxp, dxm, c2, dcp, dcm, ax, cx;
  //
  // J-Direction
  //
  // midjac = 0.5*(jac + 1.0/vol[grid_idx_p1_j]);
  midjac = 1.0/(0.5*(vol[grid_idx] + vol[grid_idx_p1_j]));
  k_x_p_half = Sj[grid_idx_p1_j].x*midjac;
  k_y_p_half = Sj[grid_idx_p1_j].y*midjac;

  // midjac = 0.5*(jac + 1.0/vol[grid_idx_m1_j]);
  midjac = 1.0/(0.5*(vol[grid_idx] + vol[grid_idx_m1_j]));
  k_x_m_half = Sj[grid_idx_p1_j].x*midjac;
  k_y_m_half = Sj[grid_idx_p1_j].y*midjac;

  nu_laminar_p_half         = 0.5*(snul[sidx] + snul[sidx+jsstride]);
  nu_turbulent_tilda_p_half = 0.5*(snut[sidx] + snut[sidx+jsstride]);
  chp_p_half       = (one_plus_cb2_by_sigma)*(nu_laminar_p_half + nu_turbulent_tilda_p_half);

  nu_laminar_m_half         = 0.5*(snul[sidx] + snul[sidx-jsstride]);
  nu_turbulent_tilda_m_half = 0.5*(snut[sidx] + snut[sidx-jsstride]);
  chp_m_half       = (one_plus_cb2_by_sigma)*(nu_laminar_m_half + nu_turbulent_tilda_m_half);

  dnuhp      = snut[sidx+jsstride] - snut[sidx];
  dnuhp_m1   = snut[sidx] - snut[sidx-jsstride];

  dxp = k_x_p_half*(xi_x) + k_y_p_half*(xi_y);
  dxm = k_x_m_half*(xi_x) + k_y_m_half*(xi_y);

  c2  = (cb2_by_sigma)*(nu_lam + nu_turb);
  dcp = dxp*(chp_p_half - c2);
  dcm = dxm*(chp_m_half - c2);

  ax = fmax(dcm,0.0);
  cx = fmax(dcp,0.0);

  if(mode==0){
    rhs_turb = rhs_turb - ax*dnuhp_m1 + cx*dnuhp;
  } else {
    L_turb_j = L_turb_j + ax;
    U_turb_j = U_turb_j + cx;
    D_turb_j = D_turb_j - ax - cx;
  }

  //
  // K-Direction 
  //
  // midjac = 0.5*(jac + 1.0/vol[grid_idx_p1_k]);
  midjac = 1.0/(0.5*(vol[grid_idx] + vol[grid_idx_p1_k]));
  k_x_p_half = Sk[grid_idx_p1_k].x*midjac;
  k_y_p_half = Sk[grid_idx_p1_k].y*midjac;

  // midjac = 0.5*(jac + 1.0/vol[grid_idx_m1_k]);
  midjac = 1.0/(0.5*(vol[grid_idx] + vol[grid_idx_m1_k]));
  k_x_m_half = Sk[grid_idx_p1_k].x*midjac;
  k_y_m_half = Sk[grid_idx_p1_k].y*midjac;

  nu_laminar_p_half         = 0.5*(snul[sidx] + snul[sidx+ksstride]);
  nu_turbulent_tilda_p_half = 0.5*(snut[sidx] + snut[sidx+ksstride]);
  chp_p_half       = (one_plus_cb2_by_sigma)*(nu_laminar_p_half + nu_turbulent_tilda_p_half);

  nu_laminar_m_half         = 0.5*(snul[sidx] + snul[sidx-ksstride]);
  nu_turbulent_tilda_m_half = 0.5*(snut[sidx] + snut[sidx-ksstride]);
  chp_m_half       = (one_plus_cb2_by_sigma)*(nu_laminar_m_half + nu_turbulent_tilda_m_half);

  dnuhp      = snut[sidx+ksstride] - snut[sidx];
  dnuhp_m1   = snut[sidx] - snut[sidx-ksstride];

  dxp = k_x_p_half*((eta_x)) + k_y_p_half*((eta_y));
  dxm = k_x_m_half*((eta_x)) + k_y_m_half*((eta_y));

  c2  = (cb2_by_sigma)*(nu_lam + nu_turb);
  dcp = dxp*(chp_p_half - c2);
  dcm = dxm*(chp_m_half - c2);

  ax = fmax(dcm,0.0);
  cx = fmax(dcp,0.0);

  if(mode==0){
    rhs_turb = rhs_turb - ax*dnuhp_m1 + cx*dnuhp;
  } else {
    L_turb_k = L_turb_k + ax;
    U_turb_k = U_turb_k + cx;
    D_turb_k = D_turb_k - ax - cx;
  }


  // ***************************************************************************
  // save everything back in global memory
  //
  if(mode==0){
    rhs[soln_idx+4] = rhs_turb;
  } else {
    LDU[j + k*jtot              ] = L_turb_j;
    LDU[j + k*jtot +   jtot*ktot] = D_turb_j;
    LDU[j + k*jtot + 2*jtot*ktot] = U_turb_j;
    LDU[j + k*jtot + 3*jtot*ktot] = L_turb_k;
    LDU[j + k*jtot + 4*jtot*ktot] = D_turb_k;
    LDU[j + k*jtot + 5*jtot*ktot] = U_turb_k;
  }

}

template<int mode>
__global__ void sa_prod_dest(int jtot, int ktot, int nvar, int nghost, double* q, double* rhs, double* LDU, double* mulam, 
			     double* vorticity, double2* xy, double2* Sj, double2* Sk, double* vol, double* dt, double* reys){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  double rey = reys[blockIdx.z];

  if(j > jtot-1-2*nghost or k > ktot-1-2*nghost) return;
  j += nghost;
  k += nghost;

  q     += (blockIdx.z*jtot*ktot)*nvar;
  mulam += (blockIdx.z*jtot*ktot);
  dt    += (blockIdx.z*jtot*ktot);

  if(mode==0){
    rhs += (blockIdx.z*jtot*ktot)*nvar;
  } else {
    LDU += (blockIdx.z*jtot*ktot)*6;
  }

  int jstride = 1;
  int kstride = jtot;
  int grid_idx = j*jstride + k*kstride;
  int soln_idx = grid_idx*(nvar);

  double nu_lam  = mulam[grid_idx]/q[soln_idx];
  double nu_turb = q[soln_idx+4];

  // double2 dxy    = 0.5*(xy[grid_idx] + xy[grid_idx+jtot])-xy[j+nghost*jtot]; // vector to wall
  double2 dxy    = 0.25*(xy[grid_idx] + xy[grid_idx+jtot] +
			 xy[grid_idx+1] + xy[grid_idx+jtot+1])-0.5*(xy[j+nghost*jtot] + xy[j+1+nghost*jtot]);

  double d_rans  = sqrt(dot(dxy,dxy));
  double d       = d_rans; // default value : RANS

#ifdef USE_DES
  double d_les  = min(dot(Sj[grid_idx], Sj[grid_idx]),dot(Sk[grid_idx], Sk[grid_idx]));
  d_les = vol[grid_idx]/sqrt(d_les);
  d = d_rans - fmax(0.0, d_rans - CDES*d_les);
#endif

  double chi              = fmax(nu_turb/nu_lam, 1.0e-12);
  double chi_power_three  = chi*chi*chi;
  double cv1_power_three  = 7.1*7.1*7.1;
  double fv1              = chi_power_three*(1.0/(chi_power_three + cv1_power_three) );
  double fv2              = 1.0 - chi*(1.0/(1 + chi*fv1)); 
  double cw1              = 3.2391/rey;

  // Remaining SA MODEL TERMS
  double inv_kappa_square = 1.0/(0.41*0.41*rey);
  double cw2              = 0.3;
  double cw3_power_six    = 64.0;
  double d2               = pow(d,2);
  double inv_dist_square  = 1.0/(d2);
  double temp_var         = inv_kappa_square*inv_dist_square;
  double S_tilda          = fmax(vorticity[grid_idx] + nu_turb*fv2*temp_var,0.3*vorticity[grid_idx]);
  // double S_tilda          = vorticity[grid_idx] + nu_turb*fv2*temp_var;
  S_tilda                 = fmax(S_tilda, 1e-12);
  double Inv_S_tilda      = 1.0/(S_tilda);
  double r                = fmin(nu_turb*temp_var*Inv_S_tilda,10.0); 
  double g                = r + cw2*(pow(r,6) - r);
  double Var4             = 1.0/(pow(g,6) + cw3_power_six);
  double Var5             = pow(((1.0+cw3_power_six)*Var4),(1.0/6.0));
  double fw               = g*Var5;

  // if(mode==0 and j==DBGJ and k==DBGK){    
  //   printf("%d %d -- | %17.9e %17.9e %17.9e\n", j, k, d, d2, inv_dist_square);   
  // }    

  // SA MODEL DERIVATIVES : dFv1,dFv2,dFw,dS_tilda,dr,dg,dfw
  double dchi             = 1.0/nu_lam;
  double dfv2             = -dchi*(1.0/(1 + chi*fv1))*(1.0/(1 + chi*fv1)); 
  double d_S_tilda        = temp_var*(fv2 + nu_turb*dfv2);
  double dr               = (S_tilda - nu_turb*d_S_tilda)*temp_var*Inv_S_tilda*Inv_S_tilda;
  double dg               = dr*(1.0 + cw2*(6*pow(r,5) - 1.0));

  double dfw              = (r>10.0)? 0.0 : Var5*dg*(1.0 - pow(g,6)*Var4);
  double itmc             = 1.0;
  double pro              = CB1*S_tilda*itmc; 
  double prod             = CB1*S_tilda*nu_turb*itmc;
  double pro_jacobian     = CB1*d_S_tilda*itmc;

  double des              = (cw1*fw)*inv_dist_square*nu_turb;
  double dest             = (cw1*fw)*inv_dist_square*nu_turb*nu_turb;
  double des_jacobian     = (cw1*fw)*inv_dist_square + (cw1*dfw)*inv_dist_square*nu_turb;


  if(mode==0){

    // rhs[soln_idx+4] = (rhs[soln_idx+4] + (prod - dest))*vol[grid_idx];
    double tk1 = fmax(nu_turb*(des_jacobian - pro_jacobian),0.0);
    double tk2 = fmax(des - pro,0.0);

    // if(j==DBGJ and k==DBGK){
    //   printf("_sarhs0_ %16.8e %16.8e %16.8e\n", (prod - dest), tk1, tk2);
    // }

    rhs[soln_idx+4] = (rhs[soln_idx+4] + (prod - dest))*SASCALE;

    // rhs[soln_idx+4] = (rhs[soln_idx+4] + (prod - dest))*dt[grid_idx];

    // if(j==DBGJ and k==DBGK){
    //   printf("%d %d -- | %16.8e %16.8e\n", j, k, rhs[soln_idx+4], dt[grid_idx]);
    // }

  } else {
    // combine all diags
    double D_turb = ( LDU[j + k*jtot +   jtot*ktot] + LDU[j + k*jtot + 4*jtot*ktot] -
    		      fmax(nu_turb*(des_jacobian - pro_jacobian),0.0) - fmax(des - pro,0.0));

    // if(j==DBGJ and k==DBGK){
    //   // printf("%d %d -- | %16.8e %16.8e\n", j, k, fmax(nu_turb*(des_jacobian - pro_jacobian),0.0), fmax(des - pro,0.0));
    //   printf("%d %d -- | %16.8e %16.8e\n", j, k, LDU[j + k*jtot +   jtot*ktot], LDU[j + k*jtot + 4*jtot*ktot]);
    // }

    LDU[j + k*jtot +   jtot*ktot] = 1.0 - dt[grid_idx]*D_turb; // D_turb_j
    LDU[j + k*jtot + 4*jtot*ktot] = 1.0 - dt[grid_idx]*D_turb; // D_turb_k

    LDU[j + k*jtot              ] *= -dt[grid_idx];
    LDU[j + k*jtot + 2*jtot*ktot] *= -dt[grid_idx];
    LDU[j + k*jtot + 3*jtot*ktot] *= -dt[grid_idx];
    LDU[j + k*jtot + 5*jtot*ktot] *= -dt[grid_idx];
  }

}

#define cb1       0.1355
#define sigma     0.66666666666666667
#define cb2       0.622
#define akt       0.41
#define cw1       3.2390678
#define cw2       0.3
#define cw3       2.0
#define cv1       7.1
#define ct1       1.0
#define ct2       2.0
#define ct3       1.2
#define ct4_orig  0.5
#define ct4_coder 0.05
#define Cdes      0.65
#define fwstar    0.424
#define rcv2      0.2
#define d2min     1.e-12
#define stilim    1.e-10
#define rmax      10.0
#define chilim    1.e-12
#define fturf     0.0
#define cappa2    akt*akt
#define gammaeff  1.0

template<int mode>
__global__ void sa_source(int jtot, int ktot, int nvar, int nghost, double* q, double* rhs, double* LDU, double* mulam, 
			  double* vorticity, double2* xy, double2* Sj, double2* Sk, double* vol, double* dt, double* reys){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  double rey = reys[blockIdx.z];

  if(j > jtot-1-2*nghost or k > ktot-1-2*nghost) return;
  j += nghost;
  k += nghost;

  double chi,fv1,fv2,fv3,ft2,dchi,dfv1,dfv2,dfv3;
  double dft2,d2,stilda,r,g,fw,dstild,dr,dg,dfw,gtilde,ct4;
  double pro,des,prod,dest,dpro,ddes,tk1,tk2,r5,g6,psi;

  q         += (blockIdx.z*jtot*ktot)*nvar;
  mulam     += (blockIdx.z*jtot*ktot);
  dt        += (blockIdx.z*jtot*ktot);
  vorticity += (blockIdx.z*jtot*ktot);

  int grid_idx = j + k*jtot;
  int soln_idx = grid_idx*(nvar);

  double nu_lam  = mulam[grid_idx]/q[soln_idx];
  double nu_turb = q[soln_idx+4];

  double2 dxy    = 0.25*(xy[grid_idx] + xy[grid_idx+jtot] +
			 xy[grid_idx+1] + xy[grid_idx+jtot+1])-0.5*(xy[j+nghost*jtot] + xy[j+1+nghost*jtot]);

  if(mode==0){
    rhs += (blockIdx.z*jtot*ktot)*nvar;
  } else {
    LDU += (blockIdx.z*jtot*ktot)*6;
  }

  chi=nu_turb/nu_lam;
  chi=max(chi,chilim);
  fv1  = (chi*chi*chi)/(chi*chi*chi+cv1*cv1*cv1);
  fv2  = 1.0 - (chi/(1+fv1*chi));
  ct4  = ct4_orig;
  ft2  = 0.0;
  dchi = 1./nu_lam;
  if(chi == chilim) dchi = 0.0;
  dfv1 =  (3.*cv1*cv1*cv1)*(chi*chi)*dchi*pow((1./(chi*chi*chi+cv1*cv1*cv1)),2);
  dfv2 = -dchi/(1+fv1*chi)+chi*pow(1./(1+fv1*chi),2)*(dfv1*chi+dchi*fv1);
  dft2 = (-2.0*ct4)*chi*dchi*ft2;

  // d  = sqrt(dot(dxy,dxy));
  d2 = max(dot(dxy,dxy),d2min); // dist squared

  stilda = vorticity[grid_idx] + nu_lam/(d2*cappa2*rey)*chi*fv2;
  stilda = max(stilda,0.3*vorticity[grid_idx]);
  // stilda = max(stilda,stilim);
  r=chi*nu_lam/(d2*cappa2*rey*stilda);
  r=min(r,rmax);
  if (r > 1e-8){
    r5=pow(r,5);
    g=r*(1.+cw2*(r5-1));
    g6=pow(g,6);
  } else {
    r5=0.0;
    g=r*(1.+cw2*(r5-1.));
    g6=0.0;
  }

  fw=(1.+pow(cw3,6))/(g6+pow(cw3,6));
  fw=g*(pow(fw,1./6.));

  dstild = nu_lam*(dchi*fv2+chi*dfv2)/(d2*cappa2*rey);
  if (stilda == stilim) dstild = 0.0;

  dr = nu_lam*(dchi*stilda - chi*dstild)/(d2*cappa2*rey*stilda*stilda);
  if (r == rmax) dr = 0.0;
  dg=dr*(1.+cw2*(6.*r5-1.));
  dfw = pow((1.+pow(cw3,6))/(g6+pow(cw3,6)),1./6.);
  dfw =  dfw*dg*(1.- g6/(g6+pow(cw3,6)));

  pro  = gammaeff*cb1*stilda*(1.-ft2);
  des  = (cw1*fw-cb1/cappa2*ft2)/d2/rey;

  prod = cb1*stilda*(1.-ft2)*nu_turb;
  dest = (cw1*fw-cb1/cappa2*ft2)*nu_turb*nu_turb/d2/rey;

  dpro = pro*dstild/stilda - cb1*stilda*dft2;

  ddes = (cw1*dfw-cb1/cappa2*dft2)/d2/rey*nu_lam*chi;
  ddes = ddes + des;
  ddes = ddes*nu_turb;
  dpro = dpro*nu_turb;
  tk1=max(des*nu_turb-pro,0.0);
  tk2=max(ddes-dpro,0.0);

  if(mode==0){

    rhs[soln_idx+4] = (rhs[soln_idx+4] + (prod - dest))*SASCALE;

    // if(j==DBGJ and k==DBGK){
    //   printf("_sarhs1_ %16.8e %16.8e %16.8e\n", (prod - dest), tk1, tk2);
    // }

  } else {

    double D_turb = LDU[j+k*jtot+jtot*ktot] + LDU[j+k*jtot+4*jtot*ktot] - tk1 - tk2;
    
    LDU[j + k*jtot +   jtot*ktot] = 1.0 - dt[grid_idx]*D_turb; // D_turb_j
    LDU[j + k*jtot + 4*jtot*ktot] = 1.0 - dt[grid_idx]*D_turb; // D_turb_k

    LDU[j + k*jtot              ] *= -dt[grid_idx];
    LDU[j + k*jtot + 2*jtot*ktot] *= -dt[grid_idx];
    LDU[j + k*jtot + 3*jtot*ktot] *= -dt[grid_idx];
    LDU[j + k*jtot + 5*jtot*ktot] *= -dt[grid_idx];
  }



}



template<int dir>
__global__ void restride_s(int pts, int nvar, double* s, double* ssa){

  int i  = blockDim.x*blockIdx.x + threadIdx.x;

  s   += pts*nvar*blockIdx.z;
  ssa += pts*blockIdx.z;

  if(i<pts){
    if(dir==0){
      ssa[i] = s[i*nvar+4]/SASCALE;
    } else {
      s[i*nvar+4] = ssa[i];
    }
  }

}

__global__ void sa_reset(int pts, double* LDU){

  int i  = blockDim.x*blockIdx.x + threadIdx.x;

  LDU += (blockIdx.z*pts)*6;

  if(i<pts){
    LDU[i        ] = 0.0;
    LDU[i +   pts] = 1.0;
    LDU[i + 2*pts] = 0.0;
    LDU[i + 3*pts] = 0.0;
    LDU[i + 4*pts] = 1.0;
    LDU[i + 5*pts] = 0.0;
  }

}


__global__ void limit_dnut(int pts, int nvar, double* s, double* q){

  int i  = blockDim.x*blockIdx.x + threadIdx.x;

  q   += pts*nvar*blockIdx.z;
  s   += pts*nvar*blockIdx.z;

  if(i>=pts) return;

  double  nut = q[i*nvar+4];
  double dnut = s[i*nvar+4];

  // if(nut + dnut < 1e-10){
  //   s[i*nvar+4] = 1e-10 - nut;
  // }

  s[i*nvar+4] = max(dnut, 1e-10 - nut);

  // q[i*nvar+4] = nut + s[i*nvar+4];

}


template<int dir>
__global__ void invert_turb(int jtot, int ktot, int nghost, double* R, double* L, double* D, double* U){

  int j, k, istride, imax;
  j = 0; k = 0;

  if(dir == 0){
    k       = blockDim.x * blockIdx.x + threadIdx.x; // k is first dimension
    istride = 1;
    imax    = jtot-nghost*2;
    if(k > ktot-nghost*2-1) return;
  }
  if(dir == 1){
    j       = blockDim.x * blockIdx.x + threadIdx.x; // j is first dimension
    istride = jtot;
    imax    = ktot-nghost*2;
    if(j > jtot-nghost*2-1) return;
  }

  j += nghost;
  k += nghost;

  L += (blockIdx.z*jtot*ktot)*6;
  D += (blockIdx.z*jtot*ktot)*6;
  U += (blockIdx.z*jtot*ktot)*6;

  R += (blockIdx.z*jtot*ktot);

  int idx_start = j + k*jtot;

  L[idx_start] = 0.0;
  U[idx_start+(imax-1)*istride] = 0.0;

  // Initialize for forward sweep
  U[idx_start] = U[idx_start]/D[idx_start];
  R[idx_start] = R[idx_start]/D[idx_start];

  // Forward sweep
  for (int i = 1; i <= imax - 1; i++)
  {
    int idx    = idx_start + i*istride;
    int idx_m1 = idx - istride;

    double inv_main_diag = 1.0/(D[idx] - L[idx]*U[idx_m1]);

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

__global__ void sa_muturb(int jtot, int ktot, int nvar, double* q, double* muturb, double* mulam){

  int j  = blockDim.x*blockIdx.x + threadIdx.x;
  int k  = blockDim.y*blockIdx.y + threadIdx.y;

  if(j > jtot-1 or k > ktot-1) return;

  q      += (j + k*jtot + blockIdx.z*jtot*ktot)*nvar;
  mulam  += (j + k*jtot + blockIdx.z*jtot*ktot);
  muturb += (j + k*jtot + blockIdx.z*jtot*ktot);

  double rho      = q[0];
  double nu_tilde = q[4];
  double nu_lam   = mulam[0]/rho;

  double X    = fmax(nu_tilde/nu_lam, 1.0e-12);
  double fv1  = (X*X*X)/(X*X*X + 7.1*7.1*7.1);      

  muturb[0] = nu_tilde*fv1*rho;

  // if(j==DBGJ and k==DBGK){
  //   // printf("%d %d | %17.9e %17.9e %17.9e\n", j, k, muturb[0], nu_tilde, fv1);
  //   printf("%d %d | %17.9e %17.9e %17.9e\n", j, k, muturb[0], mulam[0], fv1);
  // }

}


void G2D::set_muturb(double* qtest){
  dim3 thr(16,16,1);
  dim3 blk;
  blk.x = (jtot-1)/thr.x+1;
  blk.y = (ktot-1)/thr.y+1;
  blk.z = nl;
  sa_muturb<<<blk,thr>>>(jtot,ktot,nvar,qtest,muturb,mulam);
}

void G2D::sa_rhs(double* qtest, double* stest){

  // int qcount = nl*jtot*ktot*nvar;
  int count = nl*jtot*ktot;

  dim3 thr(16,16,1);
  dim3 blk;
  blk.x = (jtot-1)/thr.x+1;
  blk.y = (ktot-1)/thr.y+1;
  blk.z = nl;

  int c=0;
  double* vort = &wrk[c]; c+=count;

  size_t mem = (thr.x+2) * (thr.y+2) * 2 * sizeof(double);

  double* LDU = NULL;

  // mulam and muturb should already have been set before this is called
  this->compute_vorticity(qtest, vort);

  sa_conv_diff<0><<<blk,thr,mem>>>(jtot,ktot,nvar,nghost,qtest,stest,LDU,mulam,Sj,Sk,vol,reys[GPU]);

  // for production / destruction we only do interior points
  blk.x = (jtot-1-nghost*2)/thr.x+1;
  blk.y = (ktot-1-nghost*2)/thr.y+1;
  blk.z = nl;

  // sa_prod_dest<0><<<blk,thr>>>(jtot,ktot,nvar,nghost,qtest,stest,LDU,mulam,vort,x[GPU],Sj,Sk,vol,dt,reys[GPU]);
  sa_source<0><<<blk,thr>>>(jtot,ktot,nvar,nghost,qtest,stest,LDU,mulam,vort,x[GPU],Sj,Sk,vol,dt,reys[GPU]);

  // this->sa_adi(stest);

}

void G2D::sa_adi(double* s){

  // int qcount = nl*jtot*ktot*nvar;
  int count = nl*jtot*ktot;

  dim3 thr(16,16,1);
  dim3 blk;
  blk.x = (jtot-1)/thr.x+1;
  blk.y = (ktot-1)/thr.y+1;
  blk.z = nl;

  int c=0;
  double* vort = &wrk[c]; c+=count;
  double* LDU  = &wrk[c]; c+=count*6;
  double* ssa  = &wrk[c]; c+=count;

  double* Lj   = &LDU[          0];
  double* Dj   = &LDU[  jtot*ktot];
  double* Uj   = &LDU[2*jtot*ktot];
  double* Lk   = &LDU[3*jtot*ktot];
  double* Dk   = &LDU[4*jtot*ktot];
  double* Uk   = &LDU[5*jtot*ktot];

  dim3 linthr(256,1,1);
  dim3 linblk(1,1,nl);
  linblk.x = (jtot*ktot-1)/linthr.x+1;

  sa_reset<<<linblk,linthr>>>(jtot*ktot,LDU);

  size_t mem = (thr.x+2) * (thr.y+2) * 2 * sizeof(double);

  sa_conv_diff<1><<<blk,thr,mem>>>(jtot,ktot,nvar,nghost,q[GPU],s,LDU,mulam,Sj,Sk,vol,reys[GPU]);

  // for vorticity and production/diffusion we only do the interior points
  blk.x = (jtot-1-nghost*2)/thr.x+1;
  blk.y = (ktot-1-nghost*2)/thr.y+1;
  blk.z = nl;

  this->compute_vorticity(q[GPU],vort);

  // sa_prod_dest<1><<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],s,LDU,mulam,vort,x[GPU],Sj,Sk,vol,dt,reys[GPU]);
  sa_source<1><<<blk,thr>>>(jtot,ktot,nvar,nghost,q[GPU],s,LDU,mulam,vort,x[GPU],Sj,Sk,vol,dt,reys[GPU]);

  restride_s<0><<<linblk,linthr>>>(jtot*ktot, nvar, s, ssa);

  dim3 blocks(1,1,nl), threads(32,1,1);

  //
  // Invert J-direction
  blocks.x = (ktot-2*nghost-1)/threads.x+1;
  invert_turb<0><<<blocks,threads>>>(jtot,ktot,nghost,ssa,Lj,Dj,Uj);

  //
  // Invert K-direction
  blocks.x = (jtot-2*nghost-1)/threads.x+1;
  invert_turb<1><<<blocks,threads>>>(jtot,ktot,nghost,ssa,Lk,Dk,Uk);

  restride_s<1><<<linblk,linthr>>>(jtot*ktot, nvar, s, ssa);

  // Limit delta nu, prevent from going negative
  limit_dnut<<<linblk,linthr>>>(jtot*ktot,nvar,s,q[GPU]);

  // set_muturb(this->q[GPU]);


}

