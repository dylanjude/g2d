#include "g2d.h"



void G2D::debug_print(int j, int k, int l, double* v, int nvar){

  double* h_v = new double[nvar];

  int idx = (j + k*jtot + l*jtot*ktot)*nvar;

  HANDLE_ERROR( cudaMemcpy(h_v, &v[idx], nvar*sizeof(double), cudaMemcpyDeviceToHost) );

  int nv = min(nvar,4);

  // for(int v=0; v<nvar; v++){
  for(int v=0; v<nv; v++){
    printf("%24.16e ", h_v[v]);
    // printf("%22.14e ", h_v[v]);
    // printf("%17.9e ", h_v[v]);
  }
  printf(" (%d %d %d %d) \n",j,k,l,idx);
  // printf("------------------------------\n");

}
