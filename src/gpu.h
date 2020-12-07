#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

static void HandleError(cudaError_t err, const char *file, int line){
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__))


// #include "cublas_v2.h"
// #include "constants.h"

// static void SafeCublas(cublasStatus_t stat, const char *file, int line){
//   if(stat != CUBLAS_STATUS_SUCCESS){
//     printf("CUBLAS ERROR: %d in %s at line %d\n", stat, file, line);
//     exit(EXIT_FAILURE);
//   }
// }
// #define SAFE_CUBLAS( stat ) (SafeCublas( stat, __FILE__, __LINE__));

