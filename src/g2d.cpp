#include "g2d.h"
#include <unordered_map>
#include <cstring>
#include <cstdio>
#include "gpu.h"
#include <ctime>
#include <chrono>
#include <sys/stat.h>

using namespace std;

bool G2D::vary_Re_with_Mach=false;

G2D::G2D(int nM,int nRey,int nAoa,int jtot,int ktot,int order,
	 double* machs_in,double* reys_in,double* aoas_in,double* xy,int eqns, string foilname, int gidx){

  //
  int ngpu, device;
  cudaDeviceProp prop;          
  HANDLE_ERROR(cudaGetDeviceCount(&ngpu));
  device = gidx%ngpu;
  HANDLE_ERROR(cudaSetDevice(device));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));         
  //

  printf("#\n");
  printf("# Using GPU %2d of %2d: %s [%X]\n", device, ngpu, prop.name, ((unsigned long long*)prop.uuid.bytes)[0]); 
  printf("#\n");
  printf("# %4d Machs :", nM);
  for(int i=0; i<nM; i++) printf(" %12.4e",machs_in[i]);
  printf("\n");
  printf("# %4d AoAs  :",nAoa);
  for(int i=0; i<nAoa; i++) printf(" %12.4e",aoas_in[i]);
  printf("\n");
  printf("# %4d Reys  :",nRey);
  for(int i=0; i<nRey; i++) printf(" %12.4e",reys_in[i]);
  printf("\n");
  if(eqns == EULER){
    printf("#    Eqn Set :   Euler\n");
  } else if(eqns == LAMINAR){
    printf("#    Eqn Set :   Laminar\n");
  } else {
    printf("#    Eqn Set :   Turbulent\n");
  }
  printf("#\n");

  this->foilname   = foilname;
		   
  this->order      = order;
  this->nghost     = (order>3)? 3 : 2;

  this->nM         = nM;
  this->nRey       = nRey;
  this->nAoa       = nAoa;
  this->nl         = nM*nRey*nAoa;
  // we're given the jtot,ktot of the grid, which is the number of
  // vertices. We actually want to number of interior cells, so -1
  // then plus fringes.
  this->jtot       = jtot-1+2*nghost; 
  this->ktot       = ktot-1+2*nghost;

  printf("#    CFD Dims : %d %d\n", this->jtot, this->ktot);
  printf("#\n");

  this->gmres_nkrylov = 10;
  this->resfile = new FILE*[nl];
  this->iforce  = 0;
  this->istep   = 0;

  this->eqns        = eqns;
  this->all_timeacc = false;

  res_fname    = new string[nl];
  forces_fname = new string[nl];
  cpcf_fname   = new string[nl];
  sol_fname    = new string[nl];

  res          = new double[nl];
  res0         = NULL;
  fhist        = new double[AVG_HIST*nl];

  this->machs[CPU] = new double[nl];
  this->aoas[CPU]  = new double[nl];
  this->reys[CPU]  = new double[nl];
  this->flags[CPU] = new unsigned char[nl];
  char charbuff[64];

  sprintf(charbuff, "%s_old", foilname.c_str());
  rename(foilname.c_str(),charbuff);
  mkdir(foilname.c_str(),0755);

  // Print a header in the residual file
  FILE* fid;
  string ts = Timer::timestring();

  double aoa360;

  int l=0;
  for(int ir=0; ir<nRey; ir++){
    for(int ia=0; ia<nAoa; ia++){
      for(int im=0; im<nM; im++){
	aoa360 = (aoas_in[ia] < 0)? aoas_in[ia]+360 : aoas_in[ia];
	aoas[CPU][l]  = aoa360;
	machs[CPU][l] = machs_in[im]; 
	if(vary_Re_with_Mach){
	  reys[CPU][l]  = reys_in[ir];
	} else {
	  reys[CPU][l]  = reys_in[ir]/machs_in[im];   // reynolds number based on a_inf
	}
	sprintf(charbuff, "_r%07.0f_a%05.1f_m%04.2f", reys_in[ir], aoa360, machs_in[im]);
	res_fname[l]    = foilname + "/" +foilname + charbuff + ".res";
	forces_fname[l] = foilname + "/" +foilname + charbuff + ".forces";
	cpcf_fname[l]   = foilname + "/" +foilname + charbuff + ".cpcf";
	sol_fname[l]    = foilname + "/" +foilname + charbuff + "_sol.dat";
	resfile[l]      = NULL;

	fid = fopen(res_fname[l].c_str(), "w");
	fprintf(fid, "# %s (M=%9.3f, Alpha=%9.3f, Re=%16.8e)\n", ts.c_str(), machs_in[im], aoas_in[ia], reys_in[ir]);
	fprintf(fid,"# iter lin %16s %16s %16s %14s\n", "l2[mean_flow]", "l2[turb]", "l2[all]", "time[s]");
	fclose(fid);

	fid = fopen(forces_fname[l].c_str(),"w");
	fprintf(fid,"# Cl Cd Cm (M=%9.3f, Alpha=%9.3f, Re=%16.8e)\n", machs_in[im], aoas_in[ia], reys_in[ir]);
	fclose(fid);

	for(int i=0; i<AVG_HIST; i++){
	  fhist[l*AVG_HIST+i] = 0.0;
	}

	flags[CPU][l] = 0;

	l++;
      }
    }
  }

  HANDLE_ERROR( cudaMalloc((void**)&this->flags[GPU], nl*sizeof(unsigned char)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->machs[GPU], nl*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->aoas[GPU],  nl*sizeof(double)) );
  HANDLE_ERROR( cudaMalloc((void**)&this->reys[GPU],  nl*sizeof(double)) );
  HANDLE_ERROR( cudaMemcpy(this->machs[GPU], this->machs[CPU], nl*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->aoas[GPU], this->aoas[CPU],   nl*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(this->reys[GPU], this->reys[CPU],   nl*sizeof(double), cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemset(this->flags[GPU], 0, nl) );


  this->nvar       = 5;
  this->x0         = (double2*)xy;

  this->debug_flag = 0;

  this->timer.tick(); // start the clock

  x[CPU] = NULL;
  x[GPU] = NULL;
  q[CPU] = NULL;
  q[GPU] = NULL;
  Sj     = NULL;
  Sk     = NULL;
  vol    = NULL;
  qp     = NULL;
  qsafe  = NULL;
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

  if(q[CPU])     delete[] q[CPU];
  if(x[CPU])     delete[] x[CPU];
  if(machs[CPU]) delete[] machs[CPU];
  if(aoas[CPU])  delete[] aoas[CPU];
  if(reys[CPU])  delete[] reys[CPU];

  if(cpcf_fname)   delete[] cpcf_fname;
  if(sol_fname)    delete[] sol_fname;
  if(forces_fname) delete[] forces_fname;
  if(res_fname)    delete[] res_fname;
  if(resfile)      delete[] resfile;

  if(res)          delete[] res;
  if(res0)         delete[] res0;  
  if(fhist)        delete[] fhist;

  if(flags[GPU]) HANDLE_ERROR( cudaFree(flags[GPU]) );
  if(q[GPU])     HANDLE_ERROR( cudaFree(q[GPU]) );
  if(x[GPU])     HANDLE_ERROR( cudaFree(x[GPU]) );
  if(qp)         HANDLE_ERROR( cudaFree(qp) );
  if(qsafe)      HANDLE_ERROR( cudaFree(qsafe) );
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

  x[CPU]       = NULL;
  x[GPU]       = NULL;
  q[CPU]       = NULL;
  q[GPU]       = NULL;
  qp           = NULL;
  qsafe        = NULL;
  dt           = NULL;
  s            = NULL;
  machs[GPU]   = NULL;
  aoas[GPU]    = NULL;
  reys[GPU]    = NULL;
  machs[CPU]   = NULL;
  aoas[CPU]    = NULL;
  reys[CPU]    = NULL;
  Sj           = NULL;
  Sk           = NULL;
  vol          = NULL;
  mulam        = NULL;
  wrk          = NULL;
  xc           = NULL;

  fhist        = NULL;
  res          = NULL;
  res0         = NULL;
	       
  gmres_r      = NULL;
  gmres_Av     = NULL;
  gmres_h      = NULL;
  gmres_g      = NULL;
  gmres_v      = NULL;
  gmres_giv    = NULL;
  
  cpcf_fname   = NULL;
  sol_fname    = NULL;
  forces_fname = NULL;
  res_fname    = NULL;

  printf("# All CPU and GPU memory has been cleaned up.\n");

}
