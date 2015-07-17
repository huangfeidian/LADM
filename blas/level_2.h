
#ifndef __H_BLAS_2_H__
#define __H_BLAS_2_H__
#include "../util/stream.h"
namespace alsm
{
	//gemv
	template <DeviceType D, typename T>
	void gemv(stream<D> stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, T alpha,
		const T* A, int lda, const T* x, T beta, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void gemv<DeviceType::CPU, float>(stream<DeviceType::CPU> stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, float alpha,
		const float* A, int lda, const float* x, float beta, float* y)
	{
		CBLAS_ORDER ord = in_ord==MatrixMemOrd::COL ? CblasColMajor : CblasRowMajor;
		CBLAS_TRANSPOSE trans = in_trans==MatrixTrans::TRANSPOSE ? CblasTrans : CblasNoTrans;
		cblas_sgemv(ord, trans, m, n, alpha, A, lda, x, 1, beta, y, 1);
	}
	template<>
	__INLINE__ void gemv<DeviceType::CPU, double>(stream<DeviceType::CPU> stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, double alpha,
		const double* A, int lda, const double* x, double beta, double* y)
	{
		CBLAS_ORDER ord = in_ord == MatrixMemOrd::COL ? CblasColMajor : CblasRowMajor;
		CBLAS_TRANSPOSE trans = in_trans == MatrixTrans::TRANSPOSE ? CblasTrans : CblasNoTrans;
		cblas_dgemv(ord, trans, m, n, alpha, A, lda, x, 1, beta, y, 1);
	}
#endif
#if ALSM_USE_GPU
	template<>
	__INLINE__ void gemv<DeviceType::GPU, float>(stream<DeviceType::GPU> stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, float alpha,
		const float* A, int lda, const float* x, float beta, float* y)
	{
		float temp_alpha = alpha;
		float temp_beta = beta;
		cublasOperation_t trans = in_trans==MatrixTrans::TRANSPOSE? CUBLAS_OP_T : CUBLAS_OP_N;
		CUBLAS_CHECK_ERR(cublasSgemv(stream.local_handle, trans, m, n, &temp_alpha, A, lda, x, 1, &temp_beta, y, 1));
	}
	template<>
	__INLINE__ void gemv<DeviceType::GPU, double>(stream<DeviceType::GPU> stream, MatrixTrans in_trans, MatrixMemOrd in_ord, int m, int n, double alpha,
		const double* A, int lda, const double* x, double beta, double* y)
	{
		double temp_alpha = alpha;
		double temp_beta = beta;
		cublasOperation_t trans = in_trans == MatrixTrans::TRANSPOSE ? CUBLAS_OP_T : CUBLAS_OP_N;
		CUBLAS_CHECK_ERR(cublasDgemv(stream.local_handle, trans, m, n, &temp_alpha, A, lda, x, 1, &temp_beta, y, 1));
	}
#endif
	//spmv
	template <DeviceType D, typename T>
	__INLINE__ void spmv(stream<D> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, T alpha,
		const T* AP, const T* x, T beta, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void spmv<DeviceType::CPU, float>(stream<DeviceType::CPU> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, float alpha,
		const float* Ap, const float* x, float beta, float* y)
	{
		CBLAS_ORDER ord = in_ord==MatrixMemOrd::COL ? CblasColMajor : CblasRowMajor;
		CBLAS_UPLO lower = in_lower==MatrixFillLower::LOWER? CblasLower : CblasUpper;
		cblas_sspmv(ord, lower, n, alpha, Ap, x, 1, beta, y, 1);
	}
	template<>
	__INLINE__ void spmv<DeviceType::CPU, double>(stream<DeviceType::CPU> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, double alpha,
		const double* Ap, const double* x, double beta, double* y)
	{
		CBLAS_ORDER ord = in_ord == MatrixMemOrd::COL ? CblasColMajor : CblasRowMajor;
		CBLAS_UPLO lower = in_lower == MatrixFillLower::LOWER ? CblasLower : CblasUpper;
		cblas_dspmv(ord, lower, n, alpha, Ap, x, 1, beta, y, 1);
	}
#endif
#if ALSM_USE_GPU
	template<>
	__INLINE__ void spmv<DeviceType::GPU, float>(stream<DeviceType::GPU> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, float alpha,
		const float* Ap, const float* x, float beta, float* y)
	{
		float temp_alpha = alpha;
		float temp_beta = beta;
		cublasFillMode_t lower = (in_lower == MatrixFillLower::LOWER ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
		CUBLAS_CHECK_ERR(cublasSspmv(stream.local_handle, lower, n, &temp_alpha, Ap, x, 1, &temp_beta, y, 1));
	}
	template<>
	__INLINE__ void spmv<DeviceType::GPU, double>(stream<DeviceType::GPU> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, double alpha,
		const double* Ap, const double* x, double beta, double* y)
	{
		double temp_alpha = alpha;
		double temp_beta = beta;
		cublasFillMode_t lower =(in_lower==MatrixFillLower::LOWER ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
		CUBLAS_CHECK_ERR(cublasDspmv(stream.local_handle, lower, n, &temp_alpha, Ap, x, 1, &temp_beta, y, 1));
	}
#endif
	//symv
	template <DeviceType D, typename T>
	void symv(stream<D> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, T alpha,
		const T* AP, int lda, const T* x, T beta, T* y);
#if ALSM_USE_CPU
	template<>
	__INLINE__ void symv<DeviceType::CPU, float>(stream<DeviceType::CPU> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, float alpha,
		const float* Ap, int lda, const float* x, float beta, float* y)
	{
		CBLAS_ORDER ord = in_ord == MatrixMemOrd::COL ? CblasColMajor : CblasRowMajor;
		CBLAS_UPLO lower = in_lower == MatrixFillLower::LOWER ? CblasLower : CblasUpper;
		cblas_ssymv(ord, lower, n, alpha, Ap, lda, x, 1, beta, y, 1);
	}
	template<>
	__INLINE__ void symv<DeviceType::CPU, double>(stream<DeviceType::CPU> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, double alpha,
		const double* Ap, int lda, const double* x, double beta, double* y)
	{
		CBLAS_ORDER ord = in_ord == MatrixMemOrd::COL ? CblasColMajor : CblasRowMajor;
		CBLAS_UPLO lower = in_lower == MatrixFillLower::LOWER ? CblasLower : CblasUpper;
		cblas_dsymv(ord, lower, n, alpha, Ap, lda, x, 1, beta, y, 1);
	}
#endif

#if ALSM_USE_GPU
	template<>
	__INLINE__ void symv<DeviceType::GPU, float>(stream<DeviceType::GPU> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, float alpha,
		const float* Ap, int lda, const float* x, float beta, float* y)
	{
		float temp_alpha = alpha;
		float temp_beta = beta;
		cublasFillMode_t lower = (in_lower == MatrixFillLower::LOWER ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
		CUBLAS_CHECK_ERR(cublasSsymv(stream.local_handle, lower, n, &temp_alpha, Ap, lda, x, 1, &temp_beta, y, 1));
	}
	template<>
	__INLINE__ void symv<DeviceType::GPU, double>(stream<DeviceType::GPU> stream, MatrixMemOrd in_ord, MatrixFillLower in_lower, int n, double alpha,
		const double* Ap, int lda, const double* x, double beta, double* y)
	{
		double temp_alpha = alpha;
		double temp_beta = beta;
		cublasFillMode_t lower = (in_lower == MatrixFillLower::LOWER ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
		CUBLAS_CHECK_ERR(cublasDsymv(stream.local_handle, lower, n, &temp_alpha, Ap, lda, x, 1, &temp_beta, y, 1));
	}
#endif
}
#endif
