#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// tri-diagonal solver
void tri(double *a, double *b, double *c, double *d, int n){

  int i;
  
  n--; // since we start from x0 (not x1)
  c[0] /= b[0];
  d[0] /= b[0];

  for (i = 1; i < n; i++) {
    c[i] /= b[i] - a[i]*c[i-1];
    d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1]);
  }

  d[n] = (d[n] - a[n]*d[n-1]) / (b[n] - a[n]*c[n-1]);

  for (i = n; i-- > 0;) {
    d[i] -= c[i]*d[i+1];
  }
}

// periodic tri-diagonal solver
void trip(double *a, double* b, double* c, double* f, int M){

  int ji;
  double x[M];
  double fdum[M];
  double z;

  memset(x,0,M*sizeof(double));
  memset(fdum,0,M*sizeof(double));

  fdum[0]   = -a[0];
  fdum[M-2] = -c[M-2];
  x[0]      = c[0]/b[0];
  f[0]      = f[0]/b[0];
  fdum[0]   = fdum[0]/b[0];

  // forward sweep for two systems
  // Note: can simplify fdum since all zero initially from ji=2:m-2
  for(ji=1; ji<M-2; ji++){
    z        = 1.0 /( b[ji] - a[ji]*x[ji-1] );
    x[ji]    = c[ji]*z;
    f[ji]    = ( f[ji] - a[ji]*f[ji-1] )*z;
    fdum[ji] = - a[ji]*fdum[ji-1]*z;
  }

  // need to include since fdum[M-2] not initially
  ji       = M-2;
  z        = 1.0 /( b[ji] - a[ji]*x[ji-1] );
  x[ji]    = c[ji]*z;
  f[ji]    = ( f[ji] - a[ji]*f[ji-1] )*z;
  fdum[ji] = ( fdum[ji] - a[ji]*fdum[ji-1] )*z;


  // backward sweep for two systems   
  for(ji=M-3; ji >=0; ji--){
    f[ji]    = f[ji]- x[ji]*f[ji+1];
    fdum[ji] = fdum[ji]- x[ji]*fdum[ji+1];
  }

  // now recombine the two solutions
  // solve at the last point
  z      = 1.0 / ( b[M-1]+a[M-1]*fdum[M-2]+c[M-1]*fdum[0] );
  f[M-1] = ( f[M-1]-a[M-1]*f[M-2]-c[M-1]*f[0] )*z;

  // now solve the rest of the points
  for(ji=0;ji<M-1;ji++){
    f[ji] = f[ji] + fdum[ji]*f[M-1];
  }

}


void middlecoff_PQ(double *P1D, double *Q1D, double *x1D, double *y1D, int jtot, int ktot){

  double A1, A3, iJ, x_eta,y_eta,x_xi,y_xi;
  double x_xi_xi, y_xi_xi, x_eta_eta, y_eta_eta;

  int j, k;

  double (*x)[jtot]      = (double (*)[jtot])x1D;
  double (*y)[jtot]      = (double (*)[jtot])y1D;

  double (*P)[jtot] = (double (*)[jtot])P1D;
  double (*Q)[jtot] = (double (*)[jtot])Q1D;

  double jfac, kfac;

  // P is phi*(...)
  // Q is psi*(...)

  //
  // J=0, JMAX boundary psi (xi constant)
  //
  for(k=1;k<ktot-1;k++){
    for(j=0;j<jtot;j+=jtot-1){
      x_eta     = (x[k+1][j]-x[k-1][j])/2.0;
      y_eta     = (y[k+1][j]-y[k-1][j])/2.0;
	
      x_eta_eta = x[k+1][j]-2.0*x[k][j]+x[k-1][j];
      y_eta_eta = y[k+1][j]-2.0*y[k][j]+y[k-1][j];

      if(fabs(y_eta) > fabs(x_eta))
	Q[k][j] = -y_eta_eta/y_eta;
      else
	Q[k][j] = -x_eta_eta/x_eta;
    }
  }
  //
  // K=0, KMAX boundary phi (eta constant)
  //
  for(j=1;j<jtot-1;j++){
    for(k=0;k<ktot;k+=ktot-1){
      x_xi      = (x[k][j+1]-x[k][j-1])/2.0;
      y_xi      = (y[k][j+1]-y[k][j-1])/2.0;

      x_xi_xi   = x[k][j-1]-2.0*x[k][j]+x[k][j+1];
      y_xi_xi   = y[k][j-1]-2.0*y[k][j]+y[k][j+1];

      if(fabs(y_xi) > fabs(x_xi)){
	P[k][j] = -y_xi_xi/y_xi;
      } else {
	P[k][j] = -x_xi_xi/x_xi;
      }
    }
  }
  //
  // Now interpolate
  //
  for(k=1;k<ktot-1;k++){
    for(j=1;j<jtot-1;j++){

      jfac = 1.0*(j)/(jtot-1);
      kfac = 1.0*(k)/(ktot-1);
      
      Q[k][j] = Q[k][0] + jfac*(Q[k][jtot-1] - Q[k][0]);
      P[k][j] = P[0][j] + kfac*(P[ktot-1][j] - P[0][j]);
    }
  }

  
  for(k=1;k<ktot-1;k++){
    for(j=1;j<jtot-1;j++){

      x_xi      = (x[k][j+1]-x[k][j-1])/2.0;
      y_xi      = (y[k][j+1]-y[k][j-1])/2.0;

      x_eta     = (x[k+1][j]-x[k-1][j])/2.0;
      y_eta     = (y[k+1][j]-y[k-1][j])/2.0;
	
      x_xi_xi   = x[k][j-1]-2.0*x[k][j]+x[k][j+1];
      y_xi_xi   = y[k][j-1]-2.0*y[k][j]+y[k][j+1];

      x_eta_eta = x[k+1][j]-2.0*x[k][j]+x[k-1][j];
      y_eta_eta = y[k+1][j]-2.0*y[k][j]+y[k-1][j];

      iJ =   1.0/(x_xi *y_eta - y_xi*x_eta );
      A1 = iJ*iJ*(x_eta*x_eta + y_eta*y_eta);
      A3 = iJ*iJ*(x_xi*x_xi   + y_xi*y_xi  );
      
      P[k][j] = A1*P[k][j];
      Q[k][j] = A3*Q[k][j];

    }
  }
}

double residual(double *x1D, double *y1D, double *rhs1D, double *P1D, double *Q1D,
	      int jtot, int ktot){

  double   (*x)[jtot]    = (double (*)[jtot]   )x1D;
  double   (*y)[jtot]    = (double (*)[jtot]   )y1D;
  double   (*P)[jtot]    = (double (*)[jtot]   )P1D;
  double   (*Q)[jtot]    = (double (*)[jtot]   )Q1D;
  double (*rhs)[jtot][2] = (double (*)[jtot][2])rhs1D;

  int j, k;
  double A1, A2, A3,x_eta,y_eta,x_xi,y_xi,J;
  double y_eta_eta, x_eta_eta, y_xi_xi, x_xi_xi;
  double L2 = 0.0;

  for(k=1;k<ktot-1;k++){
    for(j=1;j<jtot-1;j++){

      x_xi      = (x[k][j+1]-x[k][j-1])/2.0;
      y_xi      = (y[k][j+1]-y[k][j-1])/2.0;

      x_eta     = (x[k+1][j]-x[k-1][j])/2.0;
      y_eta     = (y[k+1][j]-y[k-1][j])/2.0;
	
      x_xi_xi   = x[k][j-1]-2.0*x[k][j]+x[k][j+1];
      y_xi_xi   = y[k][j-1]-2.0*y[k][j]+y[k][j+1];

      x_eta_eta = x[k+1][j]-2.0*x[k][j]+x[k-1][j];
      y_eta_eta = y[k+1][j]-2.0*y[k][j]+y[k-1][j];
	
      A1 = x_eta*x_eta + y_eta*y_eta;
      A2 = x_xi*x_eta  + y_xi*y_eta;
      A3 = x_xi*x_xi   + y_xi*y_xi;
      J  = x_xi*y_eta - y_xi*x_eta;

      rhs[k][j][0] = ( A1*(x_xi_xi) -
		       A2*(x[k+1][j+1]-x[k+1][j-1]-x[k-1][j+1]+x[k-1][j-1])/2.0 +
		       A3*(x_eta_eta) + J*J*(P[k][j]*x_xi + Q[k][j]*x_eta) );

      rhs[k][j][1] = ( A1*(y_xi_xi) -
		       A2*(y[k+1][j+1]-y[k+1][j-1]-y[k-1][j+1]+y[k-1][j-1])/2.0 +
		       A3*(y_eta_eta) + J*J*(P[k][j]*y_xi + Q[k][j]*y_eta) );

      L2 += rhs[k][j][0]*rhs[k][j][0] + rhs[k][j][1]*rhs[k][j][1];

    }
  }
  return sqrt(L2 / (2.0 * jtot * ktot));
}


void slor(double w, double *x1D, double *y1D, double *P1D, double *Q1D,
	  double *a, double *b, double *c, double *d, int jtot, int ktot){

  double   (*x)[jtot]    = (double (*)[jtot]   )x1D;
  double   (*y)[jtot]    = (double (*)[jtot]   )y1D;
  double   (*P)[jtot]    = (double (*)[jtot]   )P1D;
  double   (*Q)[jtot]    = (double (*)[jtot]   )Q1D;

  int j, jp1, jm1, k, dir;
  double A1, A2, A3,x_eta,y_eta,x_xi,y_xi,J;
  double y_eta_eta, x_eta_eta, y_xi_xi, x_xi_xi;

  for(k=1;k<ktot-1;k++){
    // loop through x and y directions
    for(dir=0; dir<2; dir++){
      //
      // Boundary Conditions
      //
      for(j=0;j<jtot-1;j++){

	// jp1 = (j+1)%(jtot-1);
	// jm1 = (j-1)%(jtot-1);
	jp1 = (j+1 > jtot-1)? 1      : j+1;
	jm1 = (j-1 < 0     )? jtot-2 : j-1;
	// jp1 = j+1;
	// jm1 = j-1;

	x_xi      = (x[k][jp1]-x[k][jm1])/2.0;
	y_xi      = (y[k][jp1]-y[k][jm1])/2.0;

	x_eta     = (x[k+1][j]-x[k-1][j])/2.0;
	y_eta     = (y[k+1][j]-y[k-1][j])/2.0;
	
	x_xi_xi   = x[k][jm1]-2.0*x[k][j]+x[k][jp1];
	y_xi_xi   = y[k][jm1]-2.0*y[k][j]+y[k][jp1];

	x_eta_eta = x[k+1][j]-2.0*x[k][j]+x[k-1][j];
	y_eta_eta = y[k+1][j]-2.0*y[k][j]+y[k-1][j];
	
	A1 = x_eta*x_eta + y_eta*y_eta;
	A2 = x_xi*x_eta  + y_xi*y_eta;
	A3 = x_xi*x_xi   + y_xi*y_xi;
	J  = x_xi*y_eta - y_xi*x_eta;

	if(dir == 0){
	  d[j] = -w*( A1*(x_xi_xi) -
		      A2*(x[k+1][jp1]-x[k+1][jm1]-x[k-1][jp1]+x[k-1][jm1])/2.0 +
		      A3*(x_eta_eta) + J*J*(P[k][j]*x_xi + Q[k][j]*x_eta) );
	} else {
	  d[j] = -w*( A1*(y_xi_xi) -
		      A2*(y[k+1][jp1]-y[k+1][jm1]-y[k-1][jp1]+y[k-1][jm1])/2.0 +
		      A3*(y_eta_eta) + J*J*(P[k][j]*y_xi + Q[k][j]*y_eta) );
	}

	a[j] = A1;
	b[j] = -2.0*A1-2.0*A3;
	c[j] = A1;

      }

      // if(k < 2){
      // 	a[0] = 0;
      // 	b[0] = 1;
      // 	c[0] = 0;
      // 	d[0] = 0;
      // }

      // a[jtot-1] = 0;
      // b[jtot-1] = 1;
      // c[jtot-1] = 0;
      // d[jtot-1] = 0;

      // trip(&a[1],&b[1],&c[1],&d[1],jtot-1);
      trip(a,b,c,d,jtot-1);
      // tri(a,b,c,d,jtot);
      
      if(dir == 0){ // x-dir
	for(j=0;j<jtot;j++){
	  x[k][j] += d[j];
	}
	x[k][jtot-1] = x[k][0];
	// x[k][0] = x[k][jtot-1];
      } else {
	for(j=0;j<jtot;j++){
	  y[k][j] += d[j];
	}
	y[k][jtot-1] = y[k][0];
	// y[k][0] = y[k][jtot-1];
      }
    }
  }
}

