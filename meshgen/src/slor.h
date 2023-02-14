
extern "C" {

void middlecoff_PQ(double *P1D, double *Q1D, double *x1D, double *y1D, int jtot, int ktot);

double residual(double *x1D, double *y1D, double *rhs1D, double *P1D, double *Q1D,
		int jtot, int ktot);

void slor(double w, double *x1D, double *y1D, double *P1D, double *Q1D,
	  double *a, double *b, double *c, double *d, int jtot, int ktot);
}
