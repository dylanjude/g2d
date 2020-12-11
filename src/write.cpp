#include "g2d.h"
#include <cstdio>
#include "gpu.h"

using namespace std;

#define SHOWFRINGE 1

void G2D::write_sols(){

  int nl     = nM*nRey*nAoa;
  int qcount = nl*jtot*ktot*nvar;
  int j,k,l,v,im,ir,ia;

  FILE *fid;
  char fname[30];

  if(not this->q[CPU]){
    this->q[CPU] = new double[nRey*nAoa*nM*jtot*ktot*nvar];
  }

  HANDLE_ERROR( cudaMemcpy(q[CPU], q[GPU], qcount*sizeof(double), cudaMemcpyDeviceToHost) );

  int nskip = max(nghost-SHOWFRINGE,0);

  int extra = (nghost <= SHOWFRINGE)? 0 : 1;

  l=0;
  for(ir=0; ir<nRey; ir++){
    for(ia=0; ia<nAoa; ia++){
      for(im=0; im<nM; im++){

	sprintf(fname, "sol_%02d_%02d_%02d.dat", im, ia, ir);
	fid = fopen(fname,"w");
	fprintf(fid,"VARIABLES = X,Y,RHO,RHOU,RHOV,E,TURB_TILDA\n");
	fprintf(fid,"Zone i=%d,j=%d,k=1,F=BLOCK\n",jtot-nskip*2+extra,ktot-nskip*2+extra);
	fprintf(fid,"VARLOCATION = ([1-2]=NODAL,[3-7]=CELLCENTERED)\n");

	for (k=nskip; k<ktot-nskip+extra; k++){
	  for (j=nskip; j<jtot-nskip+extra; j++){
	    fprintf(fid, "%10.20lf\n",x[CPU][j + k*jtot].x);
	  }
	}
	for (k=nskip; k<ktot-nskip+extra; k++){
	  for (j=nskip; j<jtot-nskip+extra; j++){
	    fprintf(fid, "%10.20lf\n",x[CPU][j + k*jtot].y);
	  }
	}
	for(v=0; v<nvar; v++){
	  for (k=nskip; k<ktot-1-nskip+extra; k++){
	    for (j=nskip; j<jtot-1-nskip+extra; j++){
	      double qq = q[CPU][(j + k*jtot + l*jtot*ktot)*nvar+v];
	      if(qq != qq) qq = 100000.0;
	      fprintf(fid, "%24.16e\n",qq);
	    }
	  }
	}

	fclose(fid);

	l++;
      }
    }
  }

}
