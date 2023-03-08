#include <stdio.h>
#include <string>
#include "structures.h"


void write_grid(std::string s, double* xyz, int dims[4]){

  int nvar, var, i, pts;

  //          0     1     2     3
  // dims is [ltot, ktot, jtot, 3]

  pts  = dims[0]*dims[1]*dims[2];
  nvar = dims[3];
  
  FILE *fid;

  fid = fopen(s.c_str(), "w");

  fprintf(fid, "%d %d %d\n", dims[2], dims[1], dims[0]);

  printf("Writing Var ");
  fflush(stdout);
  for(var=0; var<3; var++){
    printf(" %d ", var);
    fflush(stdout);
    for(i=0; i<pts; i++){
      fprintf(fid, "%25.16e\n", xyz[i*nvar+var]);
    }
  }
  if(nvar == 4){
    var = 3;
    for(i=0; i<pts; i++){
      fprintf(fid, "%3.0f\n", xyz[i*nvar+var]);
    }
  }
  printf("\n");

  fclose(fid);

}
