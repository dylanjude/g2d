#include "meshgen.hpp"

int MeshGen::write_to_file(std::string s){

  FILE *fid;
  int pts = dim->pts;

  try {
    fid = fopen(s.c_str(),"w");
  } catch(int error){
    printf("*** Could not open file for writing\n");
    return 1;
  }

  int count = 0;
  fprintf(fid, "%d %d\n", dim->jtot, dim->ktot);

  for(int i=0; i<pts; i++){
    fprintf(fid, "%25.16e ", x[i]);
    count++;
    if(count%2 == 0)
      fprintf(fid, "\n");
  }
  for(int i=0; i<pts; i++){
    fprintf(fid, "%25.16e ", y[i]);
    count++;
    if(count%2 == 0)
      fprintf(fid, "\n");
  }

  fclose(fid);

  printf("*** wrote grid file: %s\n", s.c_str());

  return 0;

}

void MeshGen::get_mesh(double **xy, int dims[3]){

   dims[0] = dim->ktot;
   dims[1] = dim->jtot;
   dims[2] = 2;

   xy[0] = new double[dim->pts*2];

   for(int i=0; i<dim->pts; i++){
      xy[0][i*2+0] = x[i];
      xy[0][i*2+1] = y[i];
   }

}

