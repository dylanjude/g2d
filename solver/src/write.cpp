#include "g2d.h"
#include <cstdio>
#include "gpu.h"
#include <cmath>

using namespace std;

#define SHOWFRINGE 0

static int already_written = 0;

void G2D::write_sols(){

  int qcount = nl*jtot*ktot*nvar;
  int j,k,l,v,im,ir,ia;

  FILE *fid;
  char fname[30];

  if(not this->q[CPU]){
    this->q[CPU] = new double[nl*jtot*ktot*nvar];
  }

  // if(already_written){
  //   printf("ignoring subsequent writes\n");
  //   return;
  // }
  // already_written=1;
  // HANDLE_ERROR( cudaMemcpy(q[CPU], s, qcount*sizeof(double), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaMemcpy(q[CPU], q[GPU], qcount*sizeof(double), cudaMemcpyDeviceToHost) );

  int nskip = max(nghost-SHOWFRINGE,0);

  int extra = (nghost <= SHOWFRINGE)? 0 : 1;

  for(l=0; l<nl; l++){

    fid = fopen(sol_fname[l].c_str(),"w");

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
	  if(not isfinite(qq)) qq = 100000.0;
	  fprintf(fid, "%24.16e\n",qq);
	}
      }
    }
    
    fclose(fid);
  }

}
