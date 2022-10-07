
extern "C" {

  /* void ss_PQ(double *P1D, double *Q1D, double *x1D, double *y1D, int jtot, int ktot, */
  /* 	     double ds1,double ds2); */


  void ss_PQ(double *P1D, double *Q1D, 
	     double *p, double *q, double *r, double *s,
	     double *x1D, double *y1D, 
	     int jtot, int ktot, double ds1, double ds2);

  void set_ss_coeffs(double a, double b, double c, double d);
  
}
