#ifndef STRUCTURES_H
#define STRUCTURES_H

typedef struct Dim {
  int jmax, kmax;
  int jtot, ktot, nghost;
  int pts;
  int jstride, kstride;
} Dim;

#endif
