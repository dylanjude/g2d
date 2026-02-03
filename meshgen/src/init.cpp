#include "meshgen.hpp"
#include <math.h>

void MeshGen::init(double ds, double stretch, double howlinear){

  int jstride = dim->jstride;
  int kstride = dim->kstride;
  int jtot = dim->jtot;
  int ktot = dim->ktot;
  double shift;

  howlinear = fmax(howlinear, 1e-6);
  howlinear = fmin(howlinear, 1e6);

  // printf("using k start = %d \n", kstart);

  int idx, j, k, jle, jp1, jm1;

  
  //
  // First "kstart" layers are in the normal direction. Do those
  // first:
  //
  double x_xi, y_xi, dx, dy, gamma, tmp;
  for(k=1; k<=kstart; k++){
    for(j=0; j<jtot; j++){
      idx = j*jstride + k*kstride;
      jp1 = (j+1 > jtot-1)? 1      : j+1;
      jp1 = jp1*jstride + (k-1)*kstride;
      jm1 = (j-1 < 0     )? jtot-2 : j-1;
      jm1 = jm1*jstride + (k-1)*kstride;

      x_xi     = (x[jp1]-x[jm1])/2.0;
      y_xi     = (y[jp1]-y[jm1])/2.0;
      gamma    = x_xi*x_xi   + y_xi*y_xi;
      tmp      = 1.0/sqrt(gamma);
      dx       = ds *pow(stretch,k) * (-y_xi) * tmp;
      dy       = ds *pow(stretch,k) * ( x_xi) * tmp;

      x[idx] = x[idx - kstride] + dx;
      y[idx] = y[idx - kstride] + dy;

    }
  }

  for(k=kstart+1; k<ktot; k++){
    for(j=0; j<jtot; j++){
      idx = j*jstride + k*kstride;
      x[idx] = x[(kstart)*kstride + j*jstride]; // k=kstart;
      y[idx] = y[(kstart)*kstride + j*jstride]; // k=kstart;
    }
  }

  jle = (dim->jtot - 1)/2;

  if(dim->jtot%2 == 0){
    printf("WARNING (meshgen init.cpp): JTOT should be an odd number!!\n");
    jle += 1;
  }

  // shift = -x[0]+0.5; // roughly equal to -0.5
  // shift = -0.5;
  shift = -x[0]/2;

  // center at 0 temporarily
  for(j=0; j<dim->pts; j++){
    x[j] = x[j] + shift;
  }

  double angle0, ds1;
  angle0 = atan((y[(kstart-1)*kstride]-y[0])/(x[(kstart-1)*kstride]-x[0]));

  //
  // Go along wake direction
  //
  j = 0;
  for(k=kstart+1;k<ktot;k++){
    idx = j*jstride + k*kstride;
    ds1 = ( x[idx-kstride] - x[idx - 2*kstride] )*stretch;
    x[idx] = x[idx - kstride] + ds1;
    y[idx] = y[idx - kstride] + ds1*angle0;
  }

  double xmax = sqrt(x[(ktot-1)*kstride]*x[(ktot-1)*kstride] +
                     y[(ktot-1)*kstride]*y[(ktot-1)*kstride]);

  // we want the angle from the airfoil center (not the trailing edge)
  // to the j=0 pt in the far-field:
  angle0 = atan(y[(ktot-1)*kstride]/x[(ktot-1)*kstride]);

  // printf("xmax, angle is : %f %f\n", xmax, angle0);

  // last j-plane
  j = jtot-1;
  for(k=kstart+1;k<ktot;k++){
    idx = j*jstride + k*kstride;
    x[idx] = x[k*kstride];
    y[idx] = y[k*kstride];
  }  

  //
  // Fill the far-field
  //
  k = ktot-1;
  for(j=0;j<jtot;j++){
    idx = j*jstride + k*kstride;
    x[idx] = -xmax*cos(M_PI*(jle-j)/jle + angle0);
    y[idx] = -xmax*sin(M_PI*(jle-j)/jle + angle0);
  }

  // flatten the outer circle just a bit
  k = ktot-1;
  double ratio = (xmax-0.5)/xmax;
  for(j=0;j<jtot;j++){
    idx = j*jstride + k*kstride;
    y[idx] = y[idx] * ratio;
  }

  double dx1, dy1, dr, R;
  double tmpx, tmpy, factor;
  //
  // Linear interpolation everywhere else
  //
  for(k=kstart+1;k<ktot-1;k++){

    dr  = x[k*kstride]-x[(k-1)*kstride];

    // weight each:
    factor = (1.0*k-kstart)/(1.0*ktot-1-kstart);
    factor = pow(factor, 1.0/howlinear);

    // if(k < 30){
    //   printf("%d : %f\n", k, factor);
    // }

    // for(j=1;j<jtot-1;j++){
    for(j=0;j<jtot;j++){

      idx = j*jstride + k*kstride;

      // linear:
      tmpx = x[j*jstride + (ktot-1)*kstride]-x[j*jstride + (k-1)*kstride];
      tmpy = y[j*jstride + (ktot-1)*kstride]-y[j*jstride + (k-1)*kstride];
      R = sqrt(tmpy*tmpy+tmpx*tmpx);
      dy = dr*tmpy/R;
      dx = dr*tmpx/R;

      // normal:
      jp1      = (j+1 > jtot-1)? 1      : j+1;
      jp1      = jp1*jstride + (k-1)*kstride;
      jm1      = (j-1 < 0     )? jtot-2 : j-1;
      jm1      = jm1*jstride + (k-1)*kstride;
      x_xi     = (x[jp1]-x[jm1])/2.0;
      y_xi     = (y[jp1]-y[jm1])/2.0;
      gamma    = x_xi*x_xi   + y_xi*y_xi;
      tmp      = 1.0/sqrt(gamma);
      dx1      = dr * (-y_xi) * tmp;
      dy1      = dr * ( x_xi) * tmp;

      x[idx] = x[idx-kstride] + dx*factor + dx1*(1.0-factor);
      y[idx] = y[idx-kstride] + dy*factor + dy1*(1.0-factor);

      // tmpx = x[idx - kstride] + dx;
      // tmpy = y[idx - kstride] + dy;


      // x[idx] = x[idx]*(1.0-factor) + tmpx*factor;
      // y[idx] = y[idx]*(1.0-factor) + tmpy*factor;
    }
  }

  for(j=0; j<dim->pts; j++){
    x[j] = x[j] - shift;
  }

}
