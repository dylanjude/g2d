#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// make AAA and BBB smaller for less stretching near the airfoil
static double AAA=0.40;   // normal at wall
static double BBB=0.30;   // spacing at wall
// make CCC and DDD smaller for less stretching near the far field
static double CCC=0.9;
static double DDD=0.9;

#define OMEGA  0.1
#define OMEGAP 0.1
#define OMEGAQ 0.1
#define OMEGAR 0.1
#define OMEGAS 0.1
#define PLIM 0.5
#define QLIM 0.5
#define RLIM 0.5
#define SLIM 0.5

void ss_PQ(double *P1D, double *Q1D, 
	   double *p, double *q, double *r, double *s,
	   double *x1D, double *y1D, 
	   int jtot, int ktot, double ds1, double ds2){

  double tmp, alpha, beta, gamma, iJ,x_xi,y_xi;
  double x_xi_xi, y_xi_xi, x_eta_eta, y_eta_eta;
  double x_eta_xi, y_eta_xi;
  double sn, R1, R2, R3, R4, newP, newQ;
  double newp, newq, newr, news;

  int j, k, jp1, jm1;

  double (*x)[jtot]      = (double (*)[jtot])x1D;
  double (*y)[jtot]      = (double (*)[jtot])y1D;

  double (*P)[jtot] = (double (*)[jtot])P1D;
  double (*Q)[jtot] = (double (*)[jtot])Q1D;

  double *x_eta, *y_eta;

  x_eta = (double*)malloc(jtot*sizeof(double));
  y_eta = (double*)malloc(jtot*sizeof(double));
  
  sn = ds1;

  // fill the rest of the eta derivatives
  k = 0;
  for(j=0;j<jtot;j++){

    jp1 = (j+1 > jtot-1)? 1      : j+1;
    jm1 = (j-1 < 0     )? jtot-2 : j-1;

    x_xi     = (x[k][jp1]-x[k][jm1])/2.0;
    y_xi     = (y[k][jp1]-y[k][jm1])/2.0;
    gamma    = x_xi*x_xi   + y_xi*y_xi;
    tmp      = 1.0/sqrt(gamma);
    x_eta[j] = sn * (-y_xi) * tmp;
    y_eta[j] = sn * ( x_xi) * tmp;
  }
  
  k = 0;
  for(j=0;j<jtot;j++){

    jp1 = (j+1 > jtot-1)? 1      : j+1;
    jm1 = (j-1 < 0     )? jtot-2 : j-1;

    x_xi      = 0.5*(x[k][jp1]-x[k][jm1]);
    y_xi      = 0.5*(y[k][jp1]-y[k][jm1]);

    alpha = x_eta[j]*x_eta[j] + y_eta[j]*y_eta[j];
    beta  = x_xi*x_eta[j]  + y_xi*y_eta[j];
    gamma = x_xi*x_xi   + y_xi*y_xi;
    iJ    =   1.0/(x_xi *y_eta[j] - y_xi*x_eta[j] );

    x_xi_xi   = x[k][jm1]-2.0*x[k][j]+x[k][jp1];
    y_xi_xi   = y[k][jm1]-2.0*y[k][j]+y[k][jp1];
    
    x_eta_eta = .5*(-7.0*x[0][j] + 8.0*x[1][j] - x[2][j]) - 3.0*x_eta[j];
    y_eta_eta = .5*(-7.0*y[0][j] + 8.0*y[1][j] - y[2][j]) - 3.0*y_eta[j];
    x_eta_xi  = .5*(x_eta[jp1]-x_eta[jm1]);
    y_eta_xi  = .5*(y_eta[jp1]-y_eta[jm1]);

    R1 = -(alpha*x_xi_xi -2.0*beta*x_eta_xi + gamma*x_eta_eta)*iJ*iJ;
    R2 = -(alpha*y_xi_xi -2.0*beta*y_eta_xi + gamma*y_eta_eta)*iJ*iJ;

    newp = ( y_eta[j]*R1 - x_eta[j]*R2)*iJ;
    newq = (-y_xi*R1  + x_xi*R2)*iJ;

    // p[j] += copysign( fmin(OMEGAP*abs(newp - p[j]), PLIM*fmax(abs(p[j]), 1)), newp - p[j] );
    // q[j] += copysign( fmin(OMEGAQ*abs(newq - q[j]), QLIM*fmax(abs(q[j]), 1)), newq - q[j] );
    p[j] = newp;
    q[j] = newq;
    
  }

  // done with eta-min boundary (k=0)

  k = ktot-1;
  sn = ds2;

  // j-boundaries
  // j = 0;
  // x_eta[j] = ds2x;
  // y_eta[j] = ds2y;
  // j = jtot-1;
  // x_eta[j] = ds2x;
  // y_eta[j] = ds2y;

  // fill the rest of the eta derivatives
  for(j=0;j<jtot;j++){

    jp1 = (j+1 > jtot-1)? 1      : j+1;
    jm1 = (j-1 < 0     )? jtot-2 : j-1;

    // sn       = ds2x; // because we forced this point
    // 				  // along the wake
    x_xi     = (x[k][jp1]-x[k][jm1])/2.0;
    y_xi     = (y[k][jp1]-y[k][jm1])/2.0;
    gamma    = x_xi*x_xi   + y_xi*y_xi;
    tmp      = 1.0/sqrt(gamma);
    x_eta[j] = sn * (-y_xi) * tmp;
    y_eta[j] = sn * ( x_xi) * tmp;
  }

  for(j=0;j<jtot;j++){

    jp1 = (j+1 > jtot-1)? 1      : j+1;
    jm1 = (j-1 < 0     )? jtot-2 : j-1;

    x_xi      = 0.5*(x[k][jp1]-x[k][jm1]);
    y_xi      = 0.5*(y[k][jp1]-y[k][jm1]);

    alpha = x_eta[j]*x_eta[j] + y_eta[j]*y_eta[j];
    beta  = x_xi*x_eta[j]  + y_xi*y_eta[j];
    gamma = x_xi*x_xi   + y_xi*y_xi;
    iJ    =   1.0/(x_xi *y_eta[j] - y_xi*x_eta[j] );
    
    x_xi_xi   = x[k][jm1]-2.0*x[k][j]+x[k][jp1];
    y_xi_xi   = y[k][jm1]-2.0*y[k][j]+y[k][jp1];
    
    x_eta_eta = .5*(-7.0*x[k][j] + 
		    8.0*x[k-1][j] - x[k-2][j]) + 3.0*x_eta[j];
    y_eta_eta = .5*(-7.0*y[k][j] + 
		    8.0*y[k-1][j] - y[k-2][j]) + 3.0*y_eta[j];

    x_eta_xi = 0.5*(x_eta[jp1]-x_eta[jm1]);
    y_eta_xi = 0.5*(y_eta[jp1]-y_eta[jm1]);

    R3 = -(alpha*x_xi_xi -2.0*beta*x_eta_xi + gamma*x_eta_eta)*iJ*iJ;
    R4 = -(alpha*y_xi_xi -2.0*beta*y_eta_xi + gamma*y_eta_eta)*iJ*iJ;

    newr = ( y_eta[j]*R3 - x_eta[j]*R4)*iJ;
    news = (-y_xi*R3  + x_xi*R4)*iJ;

    // r[j] += copysign( fmin(OMEGAR*abs(newr - r[j]), RLIM*fmax(abs(r[j]), 1)), newr - r[j] );
    // s[j] += copysign( fmin(OMEGAS*abs(news - s[j]), SLIM*fmax(abs(s[j]), 1)), news - s[j] );
    r[j] = newr;
    s[j] = news;

  }

  // ok now we have p,q,r,s for all j
  for(k=1; k<ktot-1; k++){
  for(j=0; j<jtot; j++){

    newP = p[j]*exp(-AAA*k) + r[j]*exp(-CCC*(ktot-1-k));
    newQ = q[j]*exp(-BBB*k) + s[j]*exp(-DDD*(ktot-1-k));

    P[k][j] = P[k][j]*(1-OMEGA) + OMEGA*newP;
    Q[k][j] = Q[k][j]*(1-OMEGA) + OMEGA*newQ;

  }
  }

  // free(p);
  // free(q);
  free(x_eta);
  free(y_eta);

}

void set_ss_coeffs(double a, double b, double c, double d){
  if(a>0) AAA=a;
  if(b>0) BBB=b;
  if(c>0) CCC=c;
  if(d>0) DDD=d;
}
                     
