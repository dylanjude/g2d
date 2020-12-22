#include "g2d.h"
#include <unordered_map>
#include <cstring>
#include <cstdio>
#include "gpu.h"
#include <ctime>
#include <chrono>

using namespace std;

G2D::G2D(int nM,int nRey,int nAoa,int jtot,int ktot,int order,double* machs,double* reys,double* aoas,double* xy,int eqns){

  printf("#\n");
  printf("# %4d Machs :", nM);
  for(int i=0; i<nM; i++) printf(" %12.4e",machs[i]);
  printf("\n");
  printf("# %4d AoAs  :",nAoa);
  for(int i=0; i<nAoa; i++) printf(" %12.4e",aoas[i]);
  printf("\n");
  printf("# %4d Reys  :",nRey);
  for(int i=0; i<nRey; i++) printf(" %12.4e",reys[i]);
  printf("\n");
  if(eqns == EULER){
    printf("#    Eqn Set :   Euler\n");
  } else if(eqns == LAMINAR){
    printf("#    Eqn Set :   Laminar\n");
  } else {
    printf("#    Eqn Set :   Turbulent\n");
  }
  printf("#\n");

  this->order  = order;
  this->nghost = (order==5)? 3 : 2;

  this->nM         = nM;
  this->nRey       = nRey;
  this->nAoa       = nAoa;
  // we're given the jtot,ktot of the grid, which is the number of
  // vertices. We actually want to number of interior cells, so -1
  // then plus fringes.
  this->jtot       = jtot-1+2*nghost; 
  this->ktot       = ktot-1+2*nghost;
  this->machs[CPU] = machs;
  this->aoas[CPU]  = aoas;
  this->reys[CPU]  = reys;

  printf("#    CFD Dims : %d %d\n", this->jtot, this->ktot);

  this->gmres_nkrylov = 10;
  this->resfile = NULL;

  this->eqns       = eqns;

  HANDLE_ERROR( cudaMalloc((void**)&this->machs[GPU],  nM*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->aoas[GPU], nAoa*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->reys[GPU], nRey*sizeof(double)) );
  HANDLE_ERROR( cudaMemcpy(this->machs[GPU], this->machs[CPU], nM*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->aoas[GPU], this->aoas[CPU], nAoa*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->reys[GPU], this->reys[CPU], nRey*sizeof(double), cudaMemcpyHostToDevice) );

  this->nvar       = 5;
  this->x0         = (double2*)xy;

  this->debug_flag = 0;

  // Print a header in the residual file
  FILE* fid;
  string ts = Timer::timestring();
  // time_t now = chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  // string ts(30, '\0');
  // strftime(&ts[0], ts.size(), "%Y-%m-%d %H:%M:%S", localtime(&now));
  fid = fopen("residuals.dat", "w");
  fprintf(fid, "# %s\n", ts.c_str());
  fprintf(fid,"# iter lin id %16s %16s %16s %14s\n", "l2[mean_flow]", "l2[turb]", "l2[all]", "time[s]");
  fclose(fid);

  this->timer.tick(); // start the clock

  x[CPU] = NULL;
  x[GPU] = NULL;
  q[CPU] = NULL;
  q[GPU] = NULL;
  Sj     = NULL;
  Sk     = NULL;
  vol    = NULL;
  qp     = NULL;
  dt     = NULL;
  mulam  = NULL;
  wrk    = NULL;
  s      = NULL;
  xc     = NULL;

  gmres_r   = NULL;
  gmres_Av  = NULL;
  gmres_h   = NULL;
  gmres_g   = NULL;
  gmres_v   = NULL;
  gmres_giv = NULL;
  gmres_scr = NULL;

}

G2D::~G2D(){

  double elapsed = timer.tock();

  printf("# Total elapsed time: %10.3f s\n", elapsed);

  if(q[CPU]) delete[] q[CPU];
  if(x[CPU]) delete[] x[CPU];

  if(q[GPU])     HANDLE_ERROR( cudaFree(q[GPU]) );
  if(x[GPU])     HANDLE_ERROR( cudaFree(x[GPU]) );
  if(qp)         HANDLE_ERROR( cudaFree(qp) );
  if(dt)         HANDLE_ERROR( cudaFree(dt) );
  if(s)          HANDLE_ERROR( cudaFree(s) );
  if(mulam)      HANDLE_ERROR( cudaFree(mulam) );
  if(wrk)        HANDLE_ERROR( cudaFree(wrk) );
  if(machs[GPU]) HANDLE_ERROR( cudaFree(machs[GPU]) );
  if(aoas[GPU])  HANDLE_ERROR( cudaFree(aoas[GPU]) );
  if(reys[GPU])  HANDLE_ERROR( cudaFree(reys[GPU]) );
  if(Sj)         HANDLE_ERROR( cudaFree(Sj) );
  if(Sk)         HANDLE_ERROR( cudaFree(Sk) );
  if(vol)        HANDLE_ERROR( cudaFree(vol) );
  if(xc)         HANDLE_ERROR( cudaFree(xc) );

  if(gmres_scr)  HANDLE_ERROR( cudaFree(gmres_scr));
  if(gmres_r  )  HANDLE_ERROR( cudaFree(gmres_r  ));
  if(gmres_Av )  HANDLE_ERROR( cudaFree(gmres_Av ));
  if(gmres_v  )  HANDLE_ERROR( cudaFree(gmres_v  ));
  if(gmres_h  )  delete[] gmres_h;
  if(gmres_g  )  delete[] gmres_g;
  if(gmres_giv)  delete[] gmres_giv;

  x[CPU]     = NULL;
  x[GPU]     = NULL;
  q[CPU]     = NULL;
  q[GPU]     = NULL;
  qp         = NULL;
  dt         = NULL;
  s          = NULL;
  machs[GPU] = NULL;
  aoas[GPU]  = NULL;
  reys[GPU]  = NULL;
  Sj         = NULL;
  Sk         = NULL;
  vol        = NULL;
  mulam      = NULL;
  wrk        = NULL;
  xc         = NULL;

  gmres_r   = NULL;
  gmres_Av  = NULL;
  gmres_h   = NULL;
  gmres_g   = NULL;
  gmres_v   = NULL;
  gmres_giv = NULL;

  printf("# All CPU and GPU memory has been cleaned up.\n");

}
