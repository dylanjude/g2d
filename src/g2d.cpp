#include "g2d.h"
#include <unordered_map>
#include <cstring>
#include <cstdio>
#include "gpu.h"

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
    printf("#  Eqn Set  :   Euler\n");
  } else if(eqns == LAMINAR){
    printf("#  Eqn Set  :   Laminar\n");
  } else {
    printf("#  Eqn Set  :   Turbulent\n");
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

  this->eqns       = eqns;


  HANDLE_ERROR( cudaMalloc((void**)&this->machs[GPU],  nM*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->aoas[GPU], nAoa*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->reys[GPU], nRey*sizeof(double)) );
  HANDLE_ERROR( cudaMemcpy(this->machs[GPU], this->machs[CPU], nM*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->aoas[GPU], this->aoas[CPU], nAoa*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->reys[GPU], this->reys[CPU], nRey*sizeof(double), cudaMemcpyHostToDevice) );

  this->nvar       = 5;
  this->x0         = (double2*)xy;

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

}

G2D::~G2D(){

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

  printf("# All CPU and GPU memory has been cleaned up.\n");

}
