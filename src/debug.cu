#include "g2d.h"



void G2D::debug_print(int j, int k, int l, double* v, int nvar){

  double* h_v = new double[nvar];

  int idx = (j + k*jtot + l*jtot*ktot)*nvar;

  HANDLE_ERROR( cudaMemcpy(h_v, &v[idx], nvar*sizeof(double), cudaMemcpyDeviceToHost) );

  printf("[%3d %3d %3d] ",j,k,l);
  for(int v=0; v<nvar; v++){
    // printf("%24.16e ", h_v[v]);
    printf("%16.8e ", h_v[v]);
  }
  printf("\n");
  // printf("------------------------------\n");

}
