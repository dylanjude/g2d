#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include "structures.h"

class MeshGen {

  double *x, *y;
  double *a, *b, *c, *d, *P, *Q;
  double *p, *q, *r, *s;
  double (*rhs)[2];
  Dim *dim;
  double omega;
  int res_freq;
  int kstart;
  
 public:
   MeshGen(double omega, int res_freq, int kstart, 
           double ds0, double stretch, double far,
           double howlinear, double* af, int jtot, int ktot);

  ~MeshGen();
  void init(double d1, double d2, double howlinear);
  int write_to_file(std::string s);
  void get_mesh(double **xy, int dims[3]);
  void poisson(int n, double omega);
};


extern "C" {
  void set_ss_coeffs(double a, double b, double c, double d);
}
