#ifndef LSMOOTH_HPP
#define LSMOOTH_HPP
#include <stdio.h>
#include <string>
#include "structures.h"

void lsmooth(double (*xyz)[3], int dims[4], double factor, bool cubic=true);
void write_grid(std::string s, double* xyz, int dims[4]);


#endif
