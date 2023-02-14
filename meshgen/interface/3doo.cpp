#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "meshgen.hpp"
#include "lsmooth.hpp"

namespace py = pybind11;

class PyMeshGen {
  MeshGen* mg=NULL;
public:
  PyMeshGen(py::array_t<double> xy, py::dict inputs){
    
    // Check validity of inputs
    if(not inputs.contains("stretch") and not inputs.contains("far")){
      printf("missing stretch or far keys\n"); throw 123;
    }
    if(not inputs.contains("ds0")){
      printf("missing ds0 key\n"); throw 123;
    }
    if(not inputs.contains("ktot")){
      printf("missing ktot key\n"); throw 123;
    }
    if(not inputs.contains("stretch"))    inputs["stretch"]    = -1.0;
    if(not inputs.contains("far"))        inputs["far"]        = -1.0;
    if(not inputs.contains("omega"))      inputs["omega"]      = 1.5;
    if(not inputs.contains("res_freq"))   inputs["res_freq"]   = 50;
    if(not inputs.contains("knormal"))    inputs["knormal"]    = 15;
    if(not inputs.contains("initlinear")) inputs["initlinear"] = 90;

    // Build Obj
    mg = new MeshGen(inputs["omega"].cast<double>(),
                     inputs["res_freq"].cast<int>(), 
                     inputs["knormal"].cast<int>(),
                     inputs["ds0"].cast<double>(), 
                     inputs["stretch"].cast<double>(), 
                     inputs["far"].cast<double>(), 
                     inputs["initlinear"].cast<double>(), 
                     (double*)xy.request().ptr,      
                     xy.size()/2,                                 // jtot
                     inputs["ktot"].cast<int>());
  };
  ~PyMeshGen(){
    if(mg) delete mg;
  }
  void poisson(int iter){
    mg->poisson(iter);
  }
  void write_to_file(std::string fname){
    mg->write_to_file(fname);
  }
  py::array_t<double> get_mesh(){
    double *xy;
    int dims[3];
    mg->get_mesh(&xy, dims);
    return py::array_t<double>(dims, xy); // copy passed back to numpy
  }
};

void pylsmooth(py::array_t<double> xyz, double factor, int cubic=1){
  int ndim  = xyz.ndim();
  if(ndim!=4){
    printf("ndim error\n");
    throw 1234;
  }
  int* dims = new int[ndim];
  for(int i=0; i<ndim; i++) dims[i] = xyz.shape(i);
  lsmooth((double(*)[3])xyz.request().ptr, dims, factor, cubic);
  delete[] dims;
}

void pywrite_grid(std::string fname, py::array_t<double> xyz){
  int ndim  = xyz.ndim();
  if(ndim!=4){
    printf("ndim error\n");
    throw 1234;
  }
  int* dims = new int[ndim];
  for(int i=0; i<ndim; i++) dims[i] = xyz.shape(i);
  write_grid(fname, (double*)xyz.request().ptr, dims);
  delete[] dims;
}

PYBIND11_MODULE(libgen2d, m) {
  m.doc() = "Lib wrapped with pybind11"; // module docstring
  py::class_<PyMeshGen>(m, "MeshGen")
    .def(py::init<py::array_t<double>,py::dict>())
    .def("write_to_file", &PyMeshGen::write_to_file)
    .def("poisson",       &PyMeshGen::poisson)
    .def("get_mesh",      &PyMeshGen::get_mesh)
    ;
  m.def("lsmooth",      &pylsmooth);
  m.def("write_grid",   &pywrite_grid);
  m.def("set_ss_coeffs",&set_ss_coeffs);
}
