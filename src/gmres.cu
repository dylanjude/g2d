#include "g2d.h"


__global__ void update_dbg(int jtot,int ktot,int nvar,int nghost, double* q, double* s){

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


void G2D::gmres(double* rhs){

  int nl     = nM*nRey*nAoa;
  int alldof = jtot*ktot*nl*nvar;
  int dof    = jtot*ktot*nvar;

  int l;

  int i,j,k;

  gmres_nkrylov=10;

  int nvec = gmres_nkrylov;

  double *norm, *residual, *tmp;
  norm     = new double[nl];
  residual = new double[nl];
  tmp      = new double[nl];

  double l2res, drop, lindrop, stmp;

  if(not gmres_scr)  HANDLE_ERROR( cudaMalloc((void**)&gmres_scr,sizeof(double)*nl*dof));
  if(not gmres_r  )  HANDLE_ERROR( cudaMalloc((void**)&gmres_r,  sizeof(double)*nl*dof));
  if(not gmres_Av )  HANDLE_ERROR( cudaMalloc((void**)&gmres_Av, sizeof(double)*nl*dof));
  if(not gmres_v  )  HANDLE_ERROR( cudaMalloc((void**)&gmres_v,  sizeof(double)*nl*dof*nvec));
  if(not gmres_h  )  gmres_h   = new double[nl*nvec*(nvec+1)];
  if(not gmres_g  )  gmres_g   = new double[nl*(nvec+1)];
  if(not gmres_giv)  gmres_giv = new double[nl*nvec*2];

  // shortcuts
  double* r   = gmres_r;
  double* Av  = gmres_Av;
  double* h   = gmres_h;
  double* g   = gmres_g;
  double* v   = gmres_v;
  double* giv = gmres_giv;
  
  double* dq = Av; // borrow

  // cudaMemset(dq, 0, nl*dof*sizeof(double));
  // cudaMemset(r,  0, nl*dof*sizeof(double));
  // this->mvp(dq,r);
  // this->l2norm(r, &stmp);
  // // printf("this should be zero: %24.16e\n", stmp);
  // stmp = -1.0;
  // this->axpy(&stmp,r,rhs,r);
  // this->l2norm(r, &stmp);
  // // printf("this should also the original residual: %24.16e\n", stmp);

  this->l2norm(rhs, norm);

  // for(l=0; l<nl; l++){
  //   printf("#  %2d %3d %24.16e\n", 0, l, norm[l]);
  // }

  for(l=0; l<nl; l++){
    g[l] = norm[l];
    for(i=1; i<nvec+1; i++){
      g[i*nl+l] = 0.0;
    }
  }

  for(l=0; l<nl; l++){
    residual[l] = norm[l];
    tmp[l]      = 1.0/norm[l];
  }

  this->axpy(tmp, rhs, NULL, &v[0]); // v[0] = rhs/norm

  for(j=0; j<nvec; j++){

    this->precondition(&v[j*dof*nl], r);

    this->mvp(r,Av);

    for(i=0; i<j+1; i++){
      this->vdp(Av,&v[i*dof*nl],&h[(j+i*nvec)*nl]);

      for(l=0; l<nl; l++){
	tmp[l] = -h[(j+i*nvec)*nl+l];
      }

      this->axpy(tmp, &v[i*dof*nl], Av, Av); // Av = Av - v*h

    }

    this->l2norm(Av,&h[((j+1)*nvec+j)*nl]);

    if(j<nvec-1){
      for(l=0; l<nl; l++){
	tmp[l] = (h[(j+(j+1)*nvec)*nl+l] > 0.0)? 1.0/h[(j+(j+1)*nvec)*nl+l] : 0.0;
      }
      this->axpy(tmp,Av,NULL,&v[(j+1)*dof*nl]); // v[j+1] = Av/h[j+1,j]
    }

    for(l=0;l<nl;l++){
      for(i=0;i<j;i++){
	stmp                   = giv[(i*2+0)*nl+l]*h[((i)*nvec+j)*nl+l] - giv[(i*2+1)*nl+l]*h[((i+1)*nvec+j)*nl+l];
	h[((i+1)*nvec+j)*nl+l] = giv[(i*2+1)*nl+l]*h[((i)*nvec+j)*nl+l] + giv[(i*2+0)*nl+l]*h[((i+1)*nvec+j)*nl+l];
	h[((  i)*nvec+j)*nl+l] = stmp;
      }

      // New rotation:
      stmp      = 1.0/sqrt(h[(j*nvec+j)*nl+l]*h[(j*nvec+j)*nl+l] + h[((j+1)*nvec+j)*nl+l]*h[((j+1)*nvec+j)*nl+l]);
      giv[(j*2+0)*nl+l] =  h[((j  )*nvec+j)*nl+l]*stmp;
      giv[(j*2+1)*nl+l] = -h[((j+1)*nvec+j)*nl+l]*stmp;
      // apply rotation to h
      h[((j  )*nvec+j)*nl+l] = giv[(j*2+0)*nl+l]*h[(j*nvec+j)*nl+l] - giv[(j*2+1)*nl+l]*h[((j+1)*nvec+j)*nl+l];
      h[((j+1)*nvec+j)*nl+l] = 0.0;
      // apply rotation to g
      stmp          = giv[(j*2+0)*nl+l]*g[j*nl+l];
      g[(j+1)*nl+l] = giv[(j*2+1)*nl+l]*g[j*nl+l];
      g[(j  )*nl+l] = stmp;

      norm[l] = fabs(g[(j+1)*nl+l]);

      if(resfile){
	double elapsed = timer.peek();
	printf("# %6d %2d %3d %16s %16s %16.8e\n", istep+1, j+1, l,  "-",  "-", norm[l]);
	fprintf(resfile, "%6d %2d %3d %16s %16s %16.8e %14.6e\n", istep+1, j+1, l,  "-1",  "-1", norm[l], elapsed);
      }
    }

  }

  j = nvec-1;

  for(i=j; i>=0; i--){
    for(l=0;l<nl;l++){
      g[i*nl+l] = g[i*nl+l]/h[(i*nvec+i)*nl+l];
      for(k=i-1;k>=0;k--){
  	g[(k)*nl+l] = g[(k)*nl+l] - g[(i)*nl+l]*h[(k*nvec+i)*nl+l];
      }
    }
  }

  this->axpy(g, &v[0], NULL, r);       // r = g[0]*v[0]

  for(i=1;i<j+1;i++){
    this->axpy(&g[i*nl], &v[i*dof*nl], r, r); // r = g[i]*v[i][icart] + r
  }

  // // check (debug)
  // double* v1 = &v[0];
  // double* v2 = &v[dof*nl];
  // this->precondition(r, v1);
  // // this->mvp(v1,v2);
  // // stmp = -1.0;
  // // this->axpy(&stmp, v2, rhs, v1); // v1 = -v2 + rhs
  // stmp = 1.0;
  // this->axpy(&stmp, v1, q[GPU], v1); // v1 = q + v1;
  // this->l2norm(v1, norm);
  // printf("check norm: %24.16e\n", norm[0]);
  // //

  this->precondition(r, rhs);

  delete[] norm;
  delete[] residual;
  delete[] tmp;
  

}
