#ifndef _H_BLAS_1_H_
#define _H_BLAS_1_H_
#include "../util/enum.h"
#include "../util/flags.h"
#include "../util/util.h"
#include "../util/stream.h"

namespace alsm
{
//copy
	template< DeviceType D, typename T> void copy(stream<D> stream, int n, const T* from_x, T* to_y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void copy<DeviceType::CPU, float >(stream<DeviceType::CPU> stream, int n, const float* from_x, float* to_y)
	{ 
		if (from_x != to_y)
		{
			cblas_scopy(n, from_x, 1, to_y, 1);
		}
		
	}
	template<>
	__INLINE__ void copy<DeviceType::CPU, double >(stream<DeviceType::CPU> stream, int n, const double* from_x, double* to_y)
	{

		if (from_x != to_y)
		{
			cblas_dcopy(n, from_x, 1, to_y, 1);
		}
	}
#endif
#if ALSM_USE_GPU
	template<>
	__INLINE__ void copy<DeviceType::GPU, float >(stream<DeviceType::GPU> stream, int n, const float* from_x, float* to_y)
	{
		if (from_x != to_y)
		{
			CUBLAS_CHECK_ERR(cublasScopy(stream.local_handle, n, from_x, 1, to_y, 1));
		}
	}
	template<>
	__INLINE__ void copy<DeviceType::GPU, double >(stream<DeviceType::GPU> stream, int n, const double* from_x, double* to_y)
	{
		if (from_x != to_y)
		{
			CUBLAS_CHECK_ERR(cublasDcopy(stream.local_handle, n, from_x, 1, to_y, 1));
		}
	}
#endif
	//swap
	template< DeviceType D, typename T> void swap(stream<D> stream, int n, T* x, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void swap<DeviceType::CPU, float >(stream<DeviceType::CPU> stream, int n, float* x, float* y)
	{
		cblas_sswap(n, x, 1, y, 1);
	}
	template<>
	__INLINE__ void swap<DeviceType::CPU, double >(stream<DeviceType::CPU> stream, int n, double* x, double* y)
	{
		cblas_dswap(n, x, 1, y, 1);
	}
#endif
#if ALSM_USE_GPU
	template<>
	__INLINE__ void swap<DeviceType::GPU, float >(stream<DeviceType::GPU> stream, int n, float* x, float* y)
	{
		CUBLAS_CHECK_ERR(cublasSswap(stream.local_handle, n, x, 1, y, 1));
	}
	template<>
	__INLINE__ void swap<DeviceType::GPU, double >(stream<DeviceType::GPU> stream, int n, double* x, double* y)
	{
		CUBLAS_CHECK_ERR(cublasDswap(stream.local_handle, n, x, 1, y, 1));
	}
#endif
	//scal
	template< DeviceType D, typename T> void scal(stream<D> stream, int n, T* x, const T* alpha);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void scal<DeviceType::CPU, float >(stream<DeviceType::CPU> stream, int n, float* x, const float* alpha)
	{
		cblas_sscal(n, *alpha, x, 1);
	}
	template<>
	__INLINE__ void scal<DeviceType::CPU, double >(stream<DeviceType::CPU> stream, int n, double* x, const double* alpha)
	{
		cblas_dscal(n, *alpha, x, 1);
	}
#endif
#if ALSM_USE_GPU
	template<>
	__INLINE__ void scal<DeviceType::GPU, float >(stream<DeviceType::GPU> stream, int n, float* x, const float* alpha)
	{
		CUBLAS_CHECK_ERR(cublasSscal(stream.local_handle, n, alpha, x, 1));
		//cudaDeviceSynchronize();
	}
	template<>
	__INLINE__ void scal<DeviceType::GPU, double >(stream<DeviceType::GPU> stream, int n, double* x, const double* alpha)
	{
		CUBLAS_CHECK_ERR(cublasDscal(stream.local_handle, n, alpha, x, 1));
	}
#endif
	// axpy
	template< DeviceType D, typename T>
	__INLINE__ void axpy(stream<D> stream, int n, const T* alpha, const T* x, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void axpy<DeviceType::CPU, float >(stream<DeviceType::CPU> stream, int n, const float* alpha, const float* x, float* y)
	{
		cblas_saxpy(n, *alpha, x, 1, y, 1);
	}
	template<>
	__INLINE__ void axpy< DeviceType::CPU, double >(stream<DeviceType::CPU> stream, int n, const double* alpha, const double* x, double* y)
	{
		cblas_daxpy(n, *alpha, x, 1, y, 1);
	}
#endif

#if ALSM_USE_GPU
	template<>
	__INLINE__ void axpy< DeviceType::GPU, float >(stream<DeviceType::GPU> stream, int n, const float* alpha, const float* x, float* y)
	{
		CUBLAS_CHECK_ERR(cublasSaxpy(stream.local_handle, n, alpha, x, 1, y, 1));
	}
	template<>
	__INLINE__ void axpy< DeviceType::GPU, double >(stream<DeviceType::GPU> stream, int n, const double* alpha, const double* x, double* y)
	{
		CUBLAS_CHECK_ERR(cublasDaxpy(stream.local_handle, n, alpha, x, 1, y, 1));
	}
#endif
	//axpby
	template< DeviceType D, typename T>
	void axpby(stream<D> stream, int n, const T* alpha, const T* x, const T*beta, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void axpby<DeviceType::CPU, float >(stream<DeviceType::CPU> stream, int n, const float* alpha, const float* x, const float* beta, float* y)
	{
		cblas_saxpby(n, *alpha, x, 1, *beta, y, 1);
	}
	template<>
	__INLINE__ void axpby< DeviceType::CPU, double >(stream<DeviceType::CPU> stream, int n, const double* alpha, const double* x, const double* beta, double* y)
	{
		cblas_daxpby(n, *alpha, x, 1, *beta, y, 1);
	}
#endif

#if ALSM_USE_GPU
	template<>
	__INLINE__ void axpby< DeviceType::GPU, float >(stream<DeviceType::GPU> stream, int n, const float* alpha, const float* x, const float* beta, float* y)
	{
		CUBLAS_CHECK_ERR(cublasSscal(stream.local_handle, n, beta, y, 1));
		CUBLAS_CHECK_ERR(cublasSaxpy(stream.local_handle, n, alpha, x, 1, y, 1));
	}
	template<>
	__INLINE__ void axpby< DeviceType::GPU, double >(stream<DeviceType::GPU> stream, int n, const double* alpha, const double* x, const double* beta, double* y)
	{
		CUBLAS_CHECK_ERR(cublasDscal(stream.local_handle, n, beta, y, 1));
		CUBLAS_CHECK_ERR(cublasDaxpy(stream.local_handle, n, alpha, x, 1, y, 1));
	}
#endif
	// dot
	template< DeviceType D, typename T> void dot(stream<D> stream, int n, const T* x, const T* y, T* result);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void dot<DeviceType::CPU, float>(stream<DeviceType::CPU> stream, int n, const float* x, const float* y, float* result)
	{
		*result = cblas_sdot(n, x, 1, y, 1);
	}
	template<>
	__INLINE__ void dot<DeviceType::CPU, double>(stream<DeviceType::CPU> stream, int n, const double* x, const double* y, double* result)
	{
		*result = cblas_ddot(n, x, 1, y, 1);
	}
#endif
#if ALSM_USE_GPU
	template<>
	__INLINE__ void dot<DeviceType::GPU, float>(stream<DeviceType::GPU> stream, int n, const float* x, const float* y, float* result)
	{
		CUBLAS_CHECK_ERR(cublasSdot(stream.local_handle, n, x, 1, y, 1, result));
	}
	template<>
	__INLINE__ void dot<DeviceType::GPU, double>(stream<DeviceType::GPU> stream, int n, const double* x, const double* y, double* result)
	{
		CUBLAS_CHECK_ERR(cublasDdot(stream.local_handle, n, x, 1, y, 1, result));
	}
#endif
	// norm 2
	template< DeviceType D, typename T> void nrm2(stream<D> stream, int n, const T* x, T* result);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void nrm2<DeviceType::CPU, float>(stream<DeviceType::CPU> stream, int n, const float* x, float* result)
	{
		*result = cblas_snrm2(n, x, 1);
	}
	template<>
	__INLINE__ void nrm2<DeviceType::CPU, double>(stream<DeviceType::CPU> stream, int n, const double* x, double* result)
	{
		*result = cblas_dnrm2(n, x, 1);
	}
#endif

#if ALSM_USE_GPU
	template<>
	__INLINE__ void nrm2<DeviceType::GPU, float>(stream<DeviceType::GPU> stream, int n, const float* x, float* result)
	{
		CUBLAS_CHECK_ERR(cublasSnrm2(stream.local_handle, n, x, 1, result));
		//cudaDeviceSynchronize();
	}
	template<>
	__INLINE__ void nrm2<DeviceType::GPU, double>(stream<DeviceType::GPU> stream, int n, const double* x, double* result)
	{
		CUBLAS_CHECK_ERR(cublasDnrm2(stream.local_handle, n, x, 1, result));
	}
#endif
	// norm 2
	template< DeviceType D, typename T> void asum(stream<D> stream, int n, const T* x, T* result);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void asum<DeviceType::CPU, float>(stream<DeviceType::CPU> stream, int n, const float* x, float* result)
	{
		*result = cblas_sasum(n, x, 1);
	}
	template<>
	__INLINE__ void asum<DeviceType::CPU, double>(stream<DeviceType::CPU> stream, int n, const double* x, double* result)
	{
		*result = cblas_dasum(n, x, 1);
	}
#endif

#if ALSM_USE_GPU
	template<>
	__INLINE__ void asum<DeviceType::GPU, float>(stream<DeviceType::GPU> stream, int n, const float* x, float* result)
	{
		CUBLAS_CHECK_ERR(cublasSasum(stream.local_handle, n, x, 1, result));
		//cudaDeviceSynchronize();
	}
	template<>
	__INLINE__ void asum<DeviceType::GPU, double>(stream<DeviceType::GPU> stream, int n, const double* x, double* result)
	{
		CUBLAS_CHECK_ERR(cublasDasum(stream.local_handle, n, x, 1, result));
	}
#endif
}
#endif