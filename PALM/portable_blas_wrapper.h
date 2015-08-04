
#ifndef __PORTABLE_BLAS_WRAPPER_H__

#define __PORTABLE_BLAS_WRAPPER_H__ 
#define BLAS_IMPLEMENTATION_MKL

// Include this file to include a standardized (cpu) blas interface.
#pragma once

//#ifndef BLAS_IMPLEMENTATION_SYSTEM
//#define BLAS_IMPLEMENTATION_SYSTEM
//#endif

#ifdef BLAS_IMPLEMENTATION_MATLAB
//#warning "Using BLAS_IMPLEMENTATION_MATLAB..."
#include "matlab_blas_wrappers.cpp"

#elif defined(BLAS_IMPLEMENTATION_SYSTEM)
//#warning "Using BLAS_IMPLEMENTATION_SYSTEM..."
#include <cblas.h>
#undef sgemm
#define sgemm cblas_sgemm
#undef slamch
#define slamch cblas_slamch
#undef ssyevx
#define ssyevx cblas_ssyevx
//#undef sgemv
//#define sgemv cblas_sgemv

#elif defined(BLAS_IMPLEMENTATION_MKL)
//#warning "Using BLAS_IMPLEMENTATION_MKL..."
#include "mkl_blas_wrappers.cpp"
//#include "mkl.h"

#elif defined(BLAS_IMPLEMENTATION_ACML)
//#warning "Using BLAS_IMPLEMENTATION_ACML..."
#include "acml_blas_wrappers.cpp"

#else
#error "Preprocessor variable BLAS_IMPLEMENTATION_* is missing!"
#endif

#endif

