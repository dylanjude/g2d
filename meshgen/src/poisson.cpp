#include "meshgen.hpp"
#include "slor.h"
#include "ss.h"


void MeshGen::poisson(int n, double omega){

  double res;

  double w;
  if(omega > 0){
    w = omega;
  } else {
    w = this->omega;
  }
  
  memset(rhs, 0, dim->pts*2*sizeof(double));
  memset(P  , 0, dim->pts*1*sizeof(double));
  memset(Q  , 0, dim->pts*1*sizeof(double));
  memset(p  , 0, dim->jtot*1*sizeof(double));
  memset(q  , 0, dim->jtot*1*sizeof(double));
  memset(r  , 0, dim->jtot*1*sizeof(double));
  memset(s  , 0, dim->jtot*1*sizeof(double));

  int kstride = dim->kstride;

  double ds1 = x[kstart*kstride]  - x[(kstart-1)*kstride];
  double ds2 = x[(dim->ktot-1)*kstride]- x[(dim->ktot-2)*kstride];
  

  for(int i=0; i<n; i++){

    // middlecoff_PQ(P, Q, x, y, dim->jtot, dim->ktot);
    ss_PQ(&P[kstart*kstride], &Q[kstart*kstride],
	  p,q,r,s,
	  &x[kstart*kstride],&y[kstart*kstride],
	  dim->jtot,dim->ktot-kstart,
	  ds1, ds2);

    if((i+1)%this->res_freq == 0){
      res = residual(&x[kstart*kstride],&y[kstart*kstride], 
		     (double*)rhs,
		     &P[kstart*kstride], &Q[kstart*kstride],
		     dim->jtot,dim->ktot-kstart);
      printf("L2 Residual %4d : %e\n", i+1, res);
      if(res != res){
	throw 234;
      }

    }
    
    slor(w, &x[kstart*kstride],&y[kstart*kstride], 
	 &P[kstart*kstride], &Q[kstart*kstride],
	 a, b, c, d, dim->jtot, dim->ktot-kstart);

  }

}
