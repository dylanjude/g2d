#ifndef PYTHON_HELPER_H
#define PYTHON_HELPER_H
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdio.h>
#include <boost/python.hpp>

#define DOUBLE_TO_NUMPY(X, BO, DIMS, DDIM, BORROWED)	                                  \
  npy_intp dims[DDIM];                                                                    \
  for(int i=0; i<DDIM; i++){								  \
    dims[i] = DIMS[i];									  \
  }											  \
  PyObject *pyObj;									  \
  pyObj = PyArray_SimpleNewFromData(DDIM, dims, NPY_DOUBLE, (void*)X);	                  \
											  \
  if(BORROWED){										  \
    BO = boost::python::object(boost::python::handle<>(boost::python::borrowed(pyObj)));  \
  } else {										  \
    PyArray_ENABLEFLAGS((PyArrayObject*)pyObj, NPY_ARRAY_OWNDATA);			  \
    BO = boost::python::object(boost::python::handle<>(pyObj));				  \
  } 											  \
                                                                                          \

#define INT_TO_NUMPY(X, BO, DIMS, DDIM, BORROWED)	                                  \
  npy_intp dims[DDIM];                                                                    \
  for(int i=0; i<DDIM; i++){								  \
    dims[i] = DIMS[i];									  \
  }											  \
  PyObject *pyObj;									  \
  pyObj = PyArray_SimpleNewFromData(DDIM, dims, NPY_INT, (void*)X);	                  \
											  \
  if(BORROWED){										  \
    BO = boost::python::object(boost::python::handle<>(boost::python::borrowed(pyObj)));  \
  } else {										  \
    PyArray_ENABLEFLAGS((PyArrayObject*)pyObj, NPY_ARRAY_OWNDATA);			  \
    BO = boost::python::object(boost::python::handle<>(pyObj));				  \
  } 											  \
                                                                                          \

// http://docs.scipy.org/doc/numpy/reference/c-api.array.html#miscellaneous


// template <typename T>
// boost::python::object array_to_numpy(T* x, int* dim, int ndim, bool borrowed);

// // boost::python::object double_array_to_numpy(double* x, int len);

// boost::python::object int_array_to_numpy(int* x, int len);

// void numpy_to_double_array(boost::python::object bo, double* x, int *len);


template <typename T>
boost::python::object array_to_numpy(T* x, int* dim, int ndim, bool borrowed){

  npy_intp dims[ndim];

  for(int i=0; i<ndim; i++){
    dims[i] = dim[i];
  }

  PyObject *pyObj;
  if(boost::is_same<T,int>::value){
    pyObj = PyArray_SimpleNewFromData(ndim, dims, NPY_INT, (void*)x);
  } else  if(boost::is_same<T,double>::value){
    pyObj = PyArray_SimpleNewFromData(ndim, dims, NPY_DOUBLE, (void*)x);
  } else {
    throw 234;
  }

  boost::python::object bo;

  if(borrowed){
    bo = boost::python::object(boost::python::handle<>(boost::python::borrowed(pyObj)));
  } else {
    PyArray_ENABLEFLAGS((PyArrayObject*)pyObj, NPY_ARRAY_OWNDATA);
    bo = boost::python::object(boost::python::handle<>(pyObj));
  } 

  return bo;
  
}

// template boost::python::object array_to_numpy<double>;
// template boost::python::object array_to_numpy<int>;

void ccompilerhelp(){
  array_to_numpy<double>(NULL, NULL, 1, true);
  array_to_numpy<int>(NULL, NULL, 1, true);
}

boost::python::object int_array_to_numpy(int* x, int len){

  npy_intp dims[1];
  dims[0] = len;

  PyObject *pyObj = PyArray_SimpleNewFromData(1, dims, NPY_INT, (void*)x);

  boost::python::object bo(boost::python::handle<>(boost::python::borrowed(pyObj)));

  return bo;

}


void numpy_to_double_array(boost::python::object bo, double* x, int *len){

  PyArrayObject *arr;
  arr = (PyArrayObject*) bo.ptr();

  int ndim;
  npy_intp *dims;

  len[0] = 1;

  ndim = PyArray_NDIM(arr);
  dims = PyArray_DIMS(arr);

  for(int i=0; i<ndim; i++){
    len[0] *= dims[i];
  }

  x = (double*)PyArray_DATA(arr);

}




#endif
