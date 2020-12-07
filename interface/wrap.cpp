#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pyb_orchard.h"

namespace py = pybind11;

PYBIND11_MODULE(orchard, m) {
  m.doc() = "Orchard library wrapped with pybind11"; // module docstring
  py::class_<PybOrchard>(m, "Orchard")
    .def(py::init<py::array_t<int>,py::array_t<double>,double,int,int,int,py::dict>())
    .def("debug",             &PybOrchard::debug)
    .def("seed" ,             &PybOrchard::seed_pts)
    .def("parse_inputs" ,     &PybOrchard::parse_inputs)
    .def("get_grid_data",     &PybOrchard::get_grid_data)
    .def("write_grid" ,       &PybOrchard::write_grid)
    .def("write_solution",    &PybOrchard::write_sol)
    .def("write_restart",     &PybOrchard::write_restart)
    .def("init_flow",         &PybOrchard::init_flow)
    .def("step",              &PybOrchard::step)
    .def("vtx_markers",       &PybOrchard::init_vtx_markers)
    .def("post_process_grid", &PybOrchard::post_process_grid)
    .def("set_accelerator",   &PybOrchard::set_accelerator)
    .def("set_grid_speed",    &PybOrchard::set_grid_speed);
}
