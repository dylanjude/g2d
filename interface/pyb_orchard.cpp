#include "pyb_orchard.h"

#define PI 3.14159265358979311600

#define INT_TYPE    0
#define DOUBLE_TYPE 1

namespace py = pybind11;

#define DO_NOT_FREE py::capsule([](){})

static context static_ctx;

context* parse_ctx(py::dict dict){

  py::dict properties = dict["properties"];
  py::dict conditions = dict["conditions"];
  py::dict solver     = dict["solver"];
  py::dict octree     = dict["octree"];
  context* ctx        = &static_ctx;
  double fsmach, alpha, beta;

  if(dict["conditions"]["obeEqnSet"].cast<std::string>().compare(0,2,"sa")==0){
    ctx->turb = true;
    ctx->turb_model = dict["conditions"]["obeEqnSet"].cast<std::string>();
    ctx->nvar = 6;
  } else {
    ctx->turb = false;
    ctx->turb_model = "None";
    ctx->nvar = 5;
  }

  ctx->gamma        = dict["properties"]["gamma"].cast<double>();  
  fsmach            = dict["conditions"]["Mach"].cast<double>();
  alpha             = dict["conditions"]["alpha"].cast<double>();
  beta              = dict["conditions"]["beta"].cast<double>();
  ctx->uinf         = fsmach*cos(alpha*PI/180)*cos(beta*PI/180);
  ctx->vinf         = fsmach*cos(alpha*PI/180)*sin(beta*PI/180);
  ctx->winf         = fsmach*sin(alpha*PI/180);
  ctx->rinf         = 1.0;
  ctx->pinf         = 1.0/ctx->gamma;
  ctx->Re           = dict["conditions"]["reyNumber"].cast<double>();
  ctx->Pr           = dict["conditions"]["prandtl"].cast<double>();
  ctx->Prtr         = dict["solver"]["prtr"].cast<double>();
  ctx->CFL          = dict["conditions"]["obecfl"].cast<double>();
  ctx->dt           = dict["conditions"]["dt"].cast<double>();
  ctx->visc         = not (dict["conditions"]["obeEqnSet"].cast<std::string>().compare("euler")==0);
  ctx->precond      = dict["solver"]["preconditioner"].cast<std::string>();

  ctx->restart_file = dict["solver"]["restart_file"].cast<std::string>();
  ctx->restart_step = dict["solver"]["restart_step"].cast<int>();

  if(dict["octree"]["bc"].cast<std::string>().compare("periodic")==0){
    ctx->bctype     = BC_PERIODIC;
  } else {
    ctx->bctype     = BC_FAR;
  }
  ctx->flux_order = dict["solver"]["fluxorder"].cast<int>();
  ctx->visc_order = dict["solver"]["viscorder"].cast<int>();
  ctx->diss_order = dict["solver"]["dissorder"].cast<int>();
  ctx->res_mod    = dict["solver"]["res_freq"].cast<int>();

  ctx->diss_coeff = dict["solver"]["disscoef"].cast<double>();
  if(dict["solver"]["timeIntegrator"].cast<std::string>().compare("bdf2")==0){
    ctx->time_order = 2;
  } else {
    ctx->time_order = 1;
  }

  ctx->nkrylov    = dict["solver"]["nkrylov"].cast<int>();

  if(dict["solver"]["linear_solver"].cast<std::string>().compare("gmres")==0){
    ctx->linear_solver = GMRES_LHS;
  } else {
    ctx->linear_solver = APPROX_LHS;
  }
  ctx->case_type  = dict["solver"]["icase"].cast<std::string>();
  ctx->run_type   = dict["solver"]["run_type"].cast<std::string>();
  return ctx;
}

PybOrchard::PybOrchard(py::array_t<int> dims, py::array_t<double> anchor, double dX, int maxlev, 
		       int octantdim, int fcomm, py::dict cx)
  : Orchard((int*)dims.request().ptr, (double*)anchor.request().ptr, dX, maxlev, octantdim, parse_ctx(cx)){
  return;
}

void PybOrchard::seed_pts(py::array_t<double> pts, bool sparse){
  Orchard::seed_pts((double (*)[3])pts.request().ptr,pts.request().size/3,sparse); // call parent method
}

void PybOrchard::init_vtx_markers(py::array_t<double> fw, int n){
  Orchard::init_vtx_markers((double(*)[5])fw.request().ptr,n); // call parent method
}

void PybOrchard::parse_inputs(py::dict inputs){
  parse_ctx(inputs);
}

py::dict cuda_array_interface(void* ptr, int size, int type){

  // check big or little endian: cast as char array and check first byte
  unsigned int x = 0x76543210;
  bool little_endian = (((char*)&x)[0] == 0x10);

  py::dict d;
  d["shape"]   = Py_BuildValue("(i)", size);
  if(little_endian){
    d["typestr"] = (type==INT_TYPE)? "<i4" : "<f8"; // this is numpy format string convention
  } else {
    d["typestr"] = (type==INT_TYPE)? ">i4" : ">f8";
  }
  d["data"]    = Py_BuildValue("(K,O)", (unsigned long)ptr, Py_False); // read only is false
  d["version"] = 2;
  return d;
}

py::dict PybOrchard::get_grid_data(){

  int *shapes;
  int **iblank;
  double **q, **src;
  int *lgidx, *gints;
  double *xinfo;
  int nglobal, nlocal, octantdim, fulldim,pts, memtype;
  int nvar = static_ctx.nvar;
  bool borrowed = true;

  py::dict dict;

  // parent function
  Orchard::get_all_dims(&nglobal, &nlocal, &octantdim);

  fulldim = octantdim + 2*NFRINGE;
  octantdim += 2;
  pts     = fulldim*fulldim*fulldim;

  iblank  = new int*[nlocal];
  q       = new double*[nlocal];
  src     = new double*[nlocal];
  lgidx   = new int[nlocal];
  gints   = new int[nglobal*2];
  xinfo   = new double[nglobal*4];

  // py::capsule free_when_done(ptr,[](void *f){delete[] f;});
  py::list l1, l2, l3, l4, l5;

  // parent function
  Orchard::get_grid_data(q,src,iblank,lgidx,gints,xinfo,&memtype);

  dict["fringe_width"] = 2;

  l1 = py::list(nlocal);
  l2 = py::list(nlocal);
  l3 = py::list(nlocal);
  l4 = py::list(nlocal);

  for(int i=0; i<nlocal; i++){
    if( memtype == GPU_MEMORY ){
      l1[i] = cuda_array_interface((void*)iblank[i], pts,      INT_TYPE);
      l2[i] = cuda_array_interface((void*)q[i],      nvar*pts, DOUBLE_TYPE);
      l3[i] = cuda_array_interface((void*)src[i],    5*pts,    DOUBLE_TYPE);      
    } else {
      l1[i] = py::array_t<int>(        {pts},    {sizeof(int)}, iblank[i], DO_NOT_FREE);
      l2[i] = py::array_t<double>({nvar*pts}, {sizeof(double)},      q[i], DO_NOT_FREE);
      l3[i] = py::array_t<double>(   {5*pts}, {sizeof(double)},    src[i], DO_NOT_FREE);
    } 
    l4[i] = Py_BuildValue("[i,i,i,i]", lgidx[i], fulldim, fulldim, fulldim);
  }
  dict["iblanking"]    = l1;
  dict["q-variables"]  = l2;
  dict["source-terms"] = l3;
  dict["qParam"]       = l4;

  if( memtype == GPU_MEMORY ){
    dict["memory-type"] = "gpu";
  } else {
    dict["memory-type"] = "cpu";
  }

  l1 = py::list(nglobal);
  l2 = py::list(nglobal);
  l3 = py::list(nglobal);
  l4 = py::list(nglobal);
  l5 = py::list(nglobal);

  for(int i=0; i<nglobal; i++){
    //                              global_idx,        level,     level_id,      proc_id
    l1[i] = Py_BuildValue("[i,i,i,i]",       i, gints[i*2+0], gints[i*2+0], gints[i*2+1]);
    l2[i] = py::array_t<double>({3}, {sizeof(double)}, &xinfo[i*4]); // no capsule... so copied
    l3[i] = xinfo[i*4+3];
    l4[i] = Py_BuildValue("[i,i,i]", 1,1,1);
    l5[i] = Py_BuildValue("[i,i,i]", octantdim, octantdim, octantdim);
  }
  dict["gridParam"] = l1;
  dict["xlo"]       = l2;
  dict["dx"]        = l3;
  dict["ilo"]       = l4;
  dict["ihi"]       = l5;

  dict["search_cb"] = py::capsule((void*)Orchard::get_search_cb(), "cbfunction");

  // everything has been passed to Python. Python (numpy) has been
  // passed ownership of xlo. Deallocate the rest:
  delete[] iblank;
  delete[] q;
  delete[] src;
  delete[] lgidx;
  delete[] gints;
  delete[] xinfo;

  return dict;

}
