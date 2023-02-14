// #include <Python.h>
// #define NO_IMPORT_ARRAY
// #define PY_ARRAY_UNIQUE_SYMBOL libgen2d_ARRAY_API
// #include <numpy/ndarrayobject.h>
#include <stdio.h>
#include <string>
#include <vector>
#include "structures.h"
#include "spline.h"
#include <cmath>

void lsmooth(double (*xyz)[3], int dims[4], double factor, bool cubic=true){

  // if(factor < 1.0){
  //   printf("Smoothing factor (second arg) should be > 1.");
  //   printf("The bigger it is, the slower the wall will change\n");
  // }

  int i, im, j, k, l;
  int jstride, kstride, lstride;
  int jtot, ktot, ltot;

  if(factor < 0.0){
    cubic = false; // usually factor < 0 means we're smoothing out the
		   // corner near the tail. this is best done with
		   // linear interpolation, not cubic.
  }

  //          0     1     2     3
  // dims is [ltot, ktot, jtot, 3]

  ltot = dims[0];
  ktot = dims[1];
  jtot = dims[2];

  jstride = 1;
  kstride = jtot;
  lstride = jtot*ktot;

  std::vector<double> vx(ltot,0.0);
  std::vector<double> vy(ltot,0.0);
  std::vector<double> vz(ltot,0.0);
  std::vector<double> vt(ltot,0.0);
  tk::spline spx;
  tk::spline spy;
  tk::spline spz;

  double dt;

  double ratio;
  
  printf("Smoothing k = ");
  fflush(stdout);

  for(k=0; k<ktot; k++){
    if(k%10==0){
      printf(" %3d ", k);
      fflush(stdout);
    }
    if(factor < 0.0 or factor > 1.0){
      ratio = 1.0;
    } else {
      ratio = (k*1.0/(ktot-1));
      // ratio = (10.0 - 15.0*ratio + 6.0*ratio*ratio)*ratio*ratio*ratio;
      // ratio = pow(ratio, factor);
      ratio = (0.5 * tanh((ratio-factor)*9) + 0.5) - (0.5 * tanh((-factor)*9) + 0.5);
      // ratio = ratio*ratio;       // ^2
      // ratio = ratio*ratio;       // ^4
      // ratio = ratio*ratio;       // ^8
    }
    for(j=0; j<jtot; j++){   
      for(l=0; l<ltot; l++){
	i  = j*jstride + k*kstride + l*lstride;
	vx[l] = xyz[i][0];
	vy[l] = xyz[i][1];
	vz[l] = xyz[i][2];
      }

      for(l=1; l<ltot; l++){
	i  = j*jstride + k*kstride + l*lstride;
	im = i - lstride;
	vt[l] = vt[l-1] + sqrt( (xyz[i][0]-xyz[im][0])*(xyz[i][0]-xyz[im][0]) +
				(xyz[i][1]-xyz[im][1])*(xyz[i][1]-xyz[im][1]) +
				(xyz[i][2]-xyz[im][2])*(xyz[i][2]-xyz[im][2]) );
      }
      spx.set_points(vt, vx, cubic);
      spy.set_points(vt, vy, cubic);
      spz.set_points(vt, vz, cubic);

      dt = vt[ltot-1]/(ltot-1);

      for(l=1; l<ltot; l++){
	i  = j*jstride + k*kstride + l*lstride;
	xyz[i][0] += ratio*(spx(dt*l) - xyz[i][0]);
	xyz[i][1] += ratio*(spy(dt*l) - xyz[i][1]);
	xyz[i][2] += ratio*(spz(dt*l) - xyz[i][2]);
      }
    }
  }

  printf("\n");

}
