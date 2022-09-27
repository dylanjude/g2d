#include "meshgen.hpp"
#include <cmath>
// #include "python_helpers.hpp"

double find_stretch(int ktot, double ds0, double far){
  double low, high, stretch;
  low  = 1.001;
  high = 1.3;
  bool found = false;
  bool error = false;
  double ffar, dm1;
  int k, n=0;
  while(found == false and error == false){
    stretch = 0.5*(low + high);
    ffar    = 0.0;
    dm1     = ds0;
    for(k=1; k<ktot; k++){
      ffar += dm1*stretch;
      dm1 = dm1*stretch;
    }
    if(std::abs(ffar-far) < 1e-5){
      found = true;
    } else if(ffar > far){
      high = stretch;
    } else {
      low  = stretch;
    }
    if(n > 1000){
      error = true;
    }
    n++;
  }

  if(error){
    printf("Could not find stretching ratio. Using 1.15\n");
    stretch = 1.15;
  } else {
    printf("Using stretch factor: %7.3f\n", stretch);
  }

  return stretch;
  
}

MeshGen::MeshGen(double omega, int res_freq, int kstart, 
                 double ds0, double stretch, double far,
                 double howlinear, double* af, 
                 int jtot, int ktot){

   this->omega    = omega;
   this->res_freq = res_freq;
   this->kstart   = kstart;
   this->dim      = new Dim;

   dim->jmax      = jtot;
   dim->kmax      = ktot;
   dim->nghost    = 0;
   dim->ktot      = dim->kmax;
   dim->jtot      = dim->jmax;
   dim->pts       = dim->ktot * dim->jtot;
   dim->jstride   = 1;
   dim->kstride   = dim->jtot;

   if(stretch < 0){
      stretch = find_stretch(dim->ktot, ds0, far);
   }

   int jstride = dim->jstride;
   int kstride = dim->kstride;

   x   = new double[dim->pts];
   y   = new double[dim->pts];
   a   = new double[dim->jtot];
   b   = new double[dim->jtot];
   c   = new double[dim->jtot];
   d   = new double[dim->jtot];
   P   = new double[dim->pts];
   Q   = new double[dim->pts];
   p   = new double[dim->jtot];
   q   = new double[dim->jtot];
   r   = new double[dim->jtot];
   s   = new double[dim->jtot];
   rhs = new double[dim->pts][2];

   int idx1;
   int k = 0;
   for(int j=0; j<=dim->jtot; j++){
      idx1 = j*jstride + k*kstride;
      x[idx1] = af[j*2 + 0];
      y[idx1] = af[j*2 + 1];
   }

   this->init(ds0, stretch, howlinear);

}

MeshGen::~MeshGen(){

   delete x;
   delete y;

   delete a;
   delete b;
   delete c;
   delete d;
   delete P;
   delete Q;
   delete p;
   delete q;
   delete r;
   delete s;

   delete rhs;

}
