
#ifndef _H_BLAS_2_H_
#define _H_BLAS_2_H_
#include "../util/stream.h"
namespace alsm
{
	//gemv
	template <DeviceType D, typename T>
	void gemv(const stream<D>& stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, const T* alpha,
		const T* A, int lda, const T* x, const T* beta, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void gemv<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, const float* alpha,
		const float* A, int lda, const float* x, const float * beta, float* y)
	{
		CBLAS_ORDER ord = static_cast<bool>(in_ord) ? CblasColMajor : CblasRowMajor;
		CBLAS_TRANSPOSE trans = static_cast<bool>(in_trans) ? CblasTrans : CblasNoTrans;
		cblas_sgemv(ord, trans, m, n, *alpha, A, lda, x, 1, *beta, y, 1);
	}
	template<>
	__INLINE__ void gemv<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, const double* alpha,
		const double* A, int lda, const double* x, const double * beta, double* y)
	{
		CBLAS_ORDER ord = static_cast<bool>(in_ord) ? CblasColMajor : CblasRowMajor;
		CBLAS_TRANSPOSE trans = static_cast<bool>(in_trans) ? CblasTrans : CblasNoTrans;
		cblas_dgemv(ord, trans, m, n, *alpha, A, lda, x, 1, *beta, y, 1);
	}
#endif
#if ALSM_USE_GPU
	template<>
	__INLINE__ void gemv<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, const float* alpha,
		const float* A, int lda, const float* x, const float * beta, float* y)
	{
		cublasOperation_t trans = static_cast<bool>(in_trans) ? CUBLAS_OP_T : CUBLAS_OP_N;
		CUBLAS_CHECK_ERR(cublasSgemv(stream.local_handle, trans, m, n, alpha, A, lda, x, 1, beta, y, 1));
	}
	template<>
	__INLINE__ void gemv<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, const double* alpha,
		const double* A, int lda, const double* x, const double * beta, double* y)
	{

		cublasOperation_t trans = static_cast<bool>(in_trans) ? CUBLAS_OP_T : CUBLAS_OP_N;
		CUBLAS_CHECK_ERR(cublasDgemv(stream.local_handle, trans, m, n, alpha, A, lda, x, 1, beta, y, 1));
	}
#endif
	//spmv
	template <DeviceType D, typename T>
	__INLINE__ void spmv(const stream<D>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const T* alpha,
		const T* AP, const T* x, const T* beta, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void spmv<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const float* alpha,
		const float* Ap, const float* x, const float * beta, float* y)
	{
		CBLAS_ORDER ord = static_cast<bool>(in_ord) ? CblasColMajor : CblasRowMajor;
		CBLAS_UPLO lower = static_cast<bool>(in_lower) ? CblasLower : CblasUpper;
		cblas_sspmv(ord, lower, n, *alpha, Ap, x, 1, *beta, y, 1);
	}
	template<>
	__INLINE__ void spmv<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const double* alpha,
		const double* Ap, const double* x, const double * beta, double* y)
	{
		CBLAS_ORDER ord = static_cast<bool>(in_ord) ? CblasColMajor : CblasRowMajor;
		CBLAS_UPLO lower = static_cast<bool>(in_lower) ? CblasLower : CblasUpper;
		cblas_dspmv(ord, lower, n, *alpha, Ap, x, 1, *beta, y, 1);
	}
#endif
#if ALSM_USE_GPU
	template<>
	__INLINE__ void spmv<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const float* alpha,
		const float* Ap, const float* x, const float * beta, float* y)
	{
		cublasFillMode_t lower = static_cast<bool>(in_lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
		CUBLAS_CHECK_ERR(cublasSspmv(stream.local_handle, lower, n, alpha, Ap, x, 1, beta, y, 1));
	}
	template<>
	__INLINE__ void spmv<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const double* alpha,
		const double* Ap, const double* x, const double * beta, double* y)
	{
		cublasFillMode_t lower = static_cast<bool>(in_lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
		CUBLAS_CHECK_ERR(cublasDspmv(stream.local_handle, lower, n, alpha, Ap, x, 1, beta, y, 1));
	}
#endif
	//symv
	template <DeviceType D, typename T>
	void symv(const stream<D>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const T* alpha,
		const T* AP, int lda, const T* x, const T* beta, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void symv<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const float* alpha,
		const float* Ap, int lda, const float* x, const float * beta, float* y)
	{
		CBLAS_ORDER ord = static_cast<bool>(in_ord) ? CblasColMajor : CblasRowMajor;
		CBLAS_UPLO lower = static_cast<bool>(in_lower) ? CblasLower : CblasUpper;
		cblas_ssymv(ord, lower, n, *alpha, Ap, lda, x, 1, *beta, y, 1);
	}
	template<>
	__INLINE__ void symv<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const double* alpha,
		const double* Ap, int lda, const double* x, const double * beta, double* y)
	{
		CBLAS_ORDER ord = static_cast<bool>(in_ord) ? CblasColMajor : CblasRowMajor;
		CBLAS_UPLO lower = static_cast<bool>(in_lower) ? CblasLower : CblasUpper;
		cblas_dsymv(ord, lower, n, *alpha, Ap, lda, x, 1, *beta, y, 1);
	}
#endif

#if ALSM_USE_GPU
	template<>
	__INLINE__ void symv<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const float* alpha,
		const float* Ap, int lda, const float* x, const float * beta, float* y)
	{
		cublasFillMode_t lower = static_cast<bool>(in_lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
		CUBLAS_CHECK_ERR(cublasSsymv(stream.local_handle, lower, n, alpha, Ap, lda, x, 1, beta, y, 1));
	}
	template<>
	__INLINE__ void symv<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, const double* alpha,
		const double* Ap, int lda, const double* x, const double * beta, double* y)
	{
		cublasFillMode_t lower = static_cast<bool>(in_lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
		CUBLAS_CHECK_ERR(cublasDsymv(stream.local_handle, lower, n, alpha, Ap, lda, x, 1, beta, y, 1));
	}
#endif
}
#endif
