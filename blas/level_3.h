#ifndef _H_BLAS_3_H_
#define _H_BLAS_3_H_
#include "../util/stream.h"

namespace alsm
{

	//gemm
	template<DeviceType Device, typename T>
	void gemm(const stream<Device>& stream, MatrixMemOrd in_mem_ord, MatrixTrans transa, MatrixTrans transb, int m, int n, int k,
		const T* alpha, const T* A, int lda, const T* B, int ldb, const T* beta, T* C, int ldc);
#if ALSM_USE_CPU
	template<> void gemm<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, MatrixMemOrd in_mem_ord, MatrixTrans in_transa, MatrixTrans in_transb, int m, int n, int k,
		const float* alpha, const float* A, int lda, const float* B, int ldb,const float* beta , float* C, int ldc)
	{
		CBLAS_ORDER ord = static_cast<bool>(in_mem_ord) ? CblasColMajor : CblasRowMajor;
		CBLAS_TRANSPOSE transa = static_cast<bool>(in_transa) ? CblasTrans : CblasNoTrans;
		CBLAS_TRANSPOSE transb = static_cast<bool>(in_transb) ? CblasTrans : CblasNoTrans;
		cblas_sgemm(ord, transa, transb, m, n, k, *alpha, A,lda, B, ldb, *beta, C, ldc);
	}
	template<> void gemm<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream,MatrixMemOrd in_mem_ord, MatrixTrans in_transa, MatrixTrans in_transb, int m, int n, int k,
		const double* alpha, const double* A,int lda, const double* B, int ldb,const double* beta, double* C,int ldc)
	{
		CBLAS_ORDER ord = static_cast<bool>(in_mem_ord) ? CblasColMajor : CblasRowMajor;
		CBLAS_TRANSPOSE transa = static_cast<bool>(in_transa) ? CblasTrans : CblasNoTrans;
		CBLAS_TRANSPOSE transb = static_cast<bool>(in_transb) ? CblasTrans : CblasNoTrans;
		cblas_dgemm(ord, transa, transb, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
	}
#endif
//#if ALSM_USE_GPU
//	template<> void gemm<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, MatrixTrans in_transa, MatrixTrans in_transb, int m, int n, int k,
//		const float* alpha, const float* A, const float* B, const float* beta, float* C)
//	{
//		cublasOperation_t transa = static_cast<bool>(in_transa) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		cublasOperation_t transb = static_cast<bool>(in_transb) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		CUBLAS_CHECK_ERR(cublasSgemm(stream.local_handle, transa, transb, m, n, k, alpha, A, k, B, n, beta, C, n));
//	}
//	template<> void gemm<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, MatrixTrans in_transa, MatrixTrans in_transb, int m, int n, int k,
//		const double* alpha, const double* A, const double* B, const double* beta, double* C)
//	{
//		cublasOperation_t transa = static_cast<bool>(in_transa) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		cublasOperation_t transb = static_cast<bool>(in_transb) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		CUBLAS_CHECK_ERR(cublasDgemm(stream.local_handle, transa, transb, m, n, k, alpha, A, k, B, n, beta, C, n));
//	}
//#endif
//	// gemm batched
//	template<DeviceType Device, typename T>
//	void gemmBatch(const stream<Device>& stream, MatrixTrans transa, MatrixTrans transb, int m, int n, int k,
//		const T* alpha, const T** A, const T** B, const T* beta, T** C, int batchCount);
//#if ALSM_USE_CPU
//	template<> void gemmBatch<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, MatrixTrans in_transa, MatrixTrans in_transb, int m, int n, int k,
//		const float* alpha, const float** A, const float** B, const float* beta, float** C, int batchCount)
//	{
//		CBLAS_TRANSPOSE transa = static_cast<bool>(in_transa) ? CblasTrans : CblasNoTrans;
//		CBLAS_TRANSPOSE transb = static_cast<bool>(in_transb) ? CblasTrans : CblasNoTrans;
//		for (int i = 0; i < batchCount; i++)
//		{
//			cblas_sgemm(CblasColMajor, transa, transb, m, n, k, *alpha, *(A + i), k, *(B + i), n, *beta, *(C + i), n);
//		}
//
//	}
//	template<> void gemmBatch<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream, MatrixTrans in_transa, MatrixTrans in_transb, int m, int n, int k,
//		const double* alpha, const double** A, const double** B, const double* beta, double** C, int batchCount)
//	{
//		CBLAS_TRANSPOSE transa = static_cast<bool>(in_transa) ? CblasTrans : CblasNoTrans;
//		CBLAS_TRANSPOSE transb = static_cast<bool>(in_transb) ? CblasTrans : CblasNoTrans;
//		for (int i = 0; i < batchCount; i++)
//		{
//			cblas_dgemm(CblasColMajor, transa, transb, m, n, k, *alpha, *(A + i), k, *(B + i), n, *beta, *(C + i), n);
//		}
//	}
//#endif
//
//#if ALSM_USE_GPU
//	template<> void gemmBatch<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, MatrixTrans in_transa, MatrixTrans in_transb, int m, int n, int k,
//		const float* alpha, const float** A, const float** B, const float* beta, float** C, int batchCount)
//	{
//		cublasOperation_t transa = static_cast<bool>(in_transa) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		cublasOperation_t transb = static_cast<bool>(in_transb) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		cublasSgemmBatched(stream.local_handle, transa, transb, m, n, k, alpha, A, k, B, n, beta, C, n, batchCount);
//	}
//	template<> void gemmBatch<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, MatrixTrans in_transa, MatrixTrans in_transb, int m, int n, int k,
//		const double* alpha, const double** A, const double** B, const double* beta, double** C, int batchCount)
//	{
//		cublasOperation_t transa = static_cast<bool>(in_transa) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		cublasOperation_t transb = static_cast<bool>(in_transb) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		CUBLAS_CHECK_ERR(cublasDgemmBatched(stream.local_handle, transa, transb, m, n, k, alpha, A, k, B, n, beta, C, n, batchCount));
//	}
//#endif
//	// syrk
//	template<DeviceType Device, typename T>
//	void syrk(const stream<Device>& stream, MatrixFillLower fill, MatrixTrans trans, int n, int k,
//		const T* alpha, const T* A, const T* beta, T* C);
//#if ALSM_USE_CPU
//	template<> void syrk<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, MatrixFillLower in_lower, MatrixTrans in_trans, int n, int k,
//		const float* alpha, const float* A, const float* beta, float* C)
//	{
//		CBLAS_UPLO lower = static_cast<bool>(in_lower) ? CblasLower : CblasUpper;
//		CBLAS_TRANSPOSE trans = static_cast<bool>(in_trans) ? CblasTrans : CblasNoTrans;
//		cblas_ssyrk(CblasColMajor, lower, trans, n, k, *alpha, A, k, *beta, C, n);
//	}
//	template<> void syrk<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream, MatrixFillLower in_lower, MatrixTrans in_trans, int n, int k,
//		const double* alpha, const double* A, const double* beta, double* C)
//	{
//		CBLAS_UPLO lower = static_cast<bool>(in_lower) ? CblasLower : CblasUpper;
//		CBLAS_TRANSPOSE trans = static_cast<bool>(in_trans) ? CblasTrans : CblasNoTrans;
//		cblas_dsyrk(CblasColMajor, lower, trans, n, k, *alpha, A, k, *beta, C, n);
//	}
//#endif
//
//#if ALSM_USE_GPU
//	template<> void syrk<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, MatrixFillLower in_lower, MatrixTrans in_trans, int n, int k,
//		const float* alpha, const float* A, const float* beta, float* C)
//	{
//		cublasFillMode_t lower = static_cast<bool>(in_lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
//		cublasOperation_t trans = static_cast<bool>(in_trans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		CUBLAS_CHECK_ERR(cublasSsyrk(stream.local_handle, lower, trans, n, k, alpha, A, k, beta, C, n));
//	}
//	template<> void syrk<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, MatrixFillLower in_lower, MatrixTrans in_trans, int n, int k,
//		const double* alpha, const double* A, const double* beta, double* C)
//	{
//		cublasFillMode_t lower = static_cast<bool>(in_lower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
//		cublasOperation_t trans = static_cast<bool>(in_trans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//		CUBLAS_CHECK_ERR(cublasDsyrk(stream.local_handle, lower, trans, n, k, alpha, A, k, beta, C, n));
//	}
//#endif
}
#endif