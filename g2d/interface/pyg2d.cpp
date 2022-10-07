#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "g2d.h"
#include <string>

namespace py = pybind11;
using namespace std;

void run(py::array_t<double> machs, py::array_t<double> reys, py::array_t<double> aoas,
         py::array_t<double> xy, string eqns, string foilname, int order, int gidx){

  int nM   = machs.size();
  int nRey = reys.size();
  int nAoa = aoas.size();

  int jtot = xy.shape(1);
  int ktot = xy.shape(0);

  int ieqns;

  if(eqns.compare("euler")==0){
    ieqns = EULER;
  } else if(eqns.compare("laminar")==0){
    ieqns = LAMINAR;
  } else {
    ieqns = TURBULENT;
  }

  G2D solver(nM,nRey,nAoa,jtot,ktot,order,
             (double*)machs.request().ptr,
             (double*)reys.request().ptr,
             (double*)aoas.request().ptr,
             (double*)xy.request().ptr,
             ieqns,foilname,gidx);
  
  solver.init();
  solver.go();
  solver.write_sols();

}

PYBIND11_MODULE(garfoil, m) {
  m.doc() = "Orchard library wrapped with pybind11"; // module docstring
  m.def("run", &run);
}

