#ifndef __H_flags_H__
#define __H_flags_H__

//#define ALSM_USE_GPU
#define ALSM_USE_MKL
#define ALSM_USE_CPU
#define ALSM_USE_OMP
#define ALSM_USE_GPU
#ifndef ALSM_USE_CPU
#define ALSM_USE_CPU 0
#else
#define ALSM_USE_CPU 1
#ifndef ALSM_USE_CBLAS
#define ALSM_USE_CBLAS 0
#else
#define ALSM_USE_CBLAS 1
#endif

#ifndef ALSM_USE_MKL
#define ALSM_USE_MKL 0
#else
#define ALSM_USE_MKL 1
#endif
#endif

#ifndef ALSM_USE_GPU
#define ALSM_USE_GPU 0
#else
#define ALSM_USE_GPU 1
#endif

/*! \brief use single precition float */
#ifndef ALSM_SINGLE_PRECISION
#define ALSM_SINGLE_PRECISION 1
#endif

// use openmp
#ifndef ALSM_USE_OMP
#define ALSM_USE_OMP 0
#else
#define ALSM_USE_OMP 1
#endif

#if ALSM_USE_OMP
#include <omp.h>
#endif

#if ALSM_USE_CBLAS
extern "C"{
#include <cblas.h>

}
#include <malloc.h>

#elif ALSM_USE_MKL
#include <mkl.h>
#include <cstdlib>
#endif
#define __INLINE__ inline
#if ALSM_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define __DEVICE__ inline __device__ __host__
#define __GLOBAL__ inline __global__
#else
#define __DEVICE__ inline 
#include <cmath>
#endif
#endif
