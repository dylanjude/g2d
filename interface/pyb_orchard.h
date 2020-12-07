#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "orchard.h"

class PybOrchard : public Orchard {
 public:
  PybOrchard(pybind11::array_t<int> dims, pybind11::array_t<double> anchor, double dX, 
	     int maxlev, int octantdim, int fcomm, pybind11::dict cx);
  void seed_pts(pybind11::array_t<double> pts, bool sparse=false);
  void init_vtx_markers(pybind11::array_t<double> fw, int n);
  void parse_inputs(pybind11::dict inputs);
  pybind11::dict get_grid_data();
};
