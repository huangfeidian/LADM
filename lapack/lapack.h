#ifndef _H_ASLM_LAPACK_H_
#define _H_ASLM_LAPACK_H_

#include "../util/alloca.h"
#include "../blas/level_1.h"
#include "../blas/level_2.h"
#include "../blas/level_3.h"

#include <type_traits>
namespace alsm
{
	template <DeviceType D,typename T>
	void gels(stream<D> in_stream,MatrixMemOrd in_mem_ord, MatrixTrans in_trans, int m, int n, const T* A, int lda, const T* b,T* x);
	//we assume  just solve one single linear equation Ax=b min(|Ax-b|_2) ,so nrhs=1,and ldb=1
	//and we dont want to mutate A
	template<>
	void gels<DeviceType::CPU, float>(stream<DeviceType::CPU> in_stream, MatrixMemOrd in_mem_ord, MatrixTrans in_trans, int m, int n, const float* A, int lda, const float* b, float* x)
	{
		float* new_b;
		new_b = alsm_malloc<DeviceType::CPU, float>(m);
		alsm_memcpy<DeviceType::CPU, float>(in_stream, new_b, b, m);
		float* new_A;
		int A_size;
		if (in_mem_ord == MatrixMemOrd::COL)
		{
			A_size = n*lda;
		}
		else
		{
			A_size = m*lda;
		}
		new_A = alsm_malloc<DeviceType::CPU, float>(A_size);
		alsm_memcpy<DeviceType::CPU, float>(in_stream, new_A, A, A_size);
		int ord = static_cast<bool>(in_mem_ord) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
		char trans = static_cast<bool>(in_trans) ?'T': 'N';
		LAPACKE_sgels(ord, trans, m, n, 1, new_A, lda, new_b, m);
		//we set the x=1/2* new_b
		alsm_memcpy<DeviceType::CPU, float>(in_stream, x, new_b, n);
		alsm_free<DeviceType::CPU, float>(new_A);
		alsm_free<DeviceType::CPU, float>(new_b);
	}
	template<>
	void gels<DeviceType::CPU, double>(stream<DeviceType::CPU> in_stream, MatrixMemOrd in_mem_ord, MatrixTrans in_trans, int m, int n, const double* A, int lda, const double* b, double* x)
	{
		double* new_b;
		new_b = alsm_malloc<DeviceType::CPU, double>(m);
		alsm_memcpy<DeviceType::CPU, double>(in_stream, new_b, b, m);
		double* new_A;
		int A_size;
		if (in_mem_ord == MatrixMemOrd::COL)
		{
			A_size = n*lda;
		}
		else
		{
			A_size = m*lda;
		}
		new_A = alsm_malloc<DeviceType::CPU, double>(A_size);
		alsm_memcpy<DeviceType::CPU, double>(in_stream, new_A, A, A_size);
		int ord = static_cast<bool>(in_mem_ord) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
		char trans = static_cast<bool>(in_trans) ? 'T' : 'N';
		LAPACKE_dgels(ord, trans, m, n, 1, new_A, lda, new_b, m);
		alsm_memcpy<DeviceType::CPU, double>(in_stream, x, new_b, n);
		alsm_free<DeviceType::CPU, double>(new_A);
		alsm_free<DeviceType::CPU, double>(new_b);
	}
	//the cuda version is to be continued
	template<DeviceType D, typename T>
	void svds_max(stream<D> in_stream, MatrixMemOrd in_mem_ord, int m, int n, const T* A, int lda, T* result);
	template<>
	void svds_max<DeviceType::CPU,float>(stream<DeviceType::CPU> in_stream, MatrixMemOrd in_mem_ord,  int m, int n, const float* A, int lda,float* result)
	{
		float* ATA;
		ATA=alsm_malloc<DeviceType::CPU,float>(n*n);
		alsm_memset<DeviceType::CPU,float>(ATA, 0, n*n);
		float tau;
		float alpha_one = 1.0;
		float alpha_zero = 0.0;
		gemm<DeviceType::CPU,float>(in_stream, in_mem_ord,MatrixTrans::TRANSPOSE, MatrixTrans::NORMAL, n, n, m, &alpha_one, A,m, A, m,&alpha_zero, ATA,n);
		float s = 2*slamch("S");
		int out_m;
		float* out_w = alsm_malloc<DeviceType::CPU, float>(n);//all the eigenvalue
		float* out_z=nullptr;//we don't need the sigular vector
		int *ifail = new int[n];
		int ord = static_cast<bool>(in_mem_ord) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
		float nonsense = 0.0;
		LAPACKE_ssyevx(ord,'N', 'I', 'U', n,ATA, n, nonsense, nonsense, n, n, s, &out_m, out_w, out_z, 1,ifail);
		*result = out_w[0];
		delete [] ifail;
		alsm_free<DeviceType::CPU,float>(ATA);
		alsm_free<DeviceType::CPU, float>(out_w);
		printf("the max sigular value square  is %lf\n", static_cast<double>(*result));
	}
	template<>
	void svds_max<DeviceType::CPU, double>(stream<DeviceType::CPU> in_stream, MatrixMemOrd in_mem_ord, int m, int n, const double* A, int lda, double* result)
	{
		double* ATA;
		ATA = alsm_malloc<DeviceType::CPU, double>(n*n);
		alsm_memset<DeviceType::CPU, double>(ATA, 0, n*n);
		double tau;
		double alpha_one = 1.0;
		double alpha_zero = 0.0;
		gemm<DeviceType::CPU, double>(in_stream, in_mem_ord, MatrixTrans::TRANSPOSE, MatrixTrans::NORMAL, n, n, m, &alpha_one, A, m, A, m, &alpha_zero, ATA, n);
		double s = 2 * slamch("S");
		int out_m;
		double* out_w = alsm_malloc<DeviceType::CPU, double>(n);//all the eigenvalue
		double* out_z = nullptr;//we don't need the sigular vector
		int *ifail = new int[n];
		int ord = static_cast<bool>(in_mem_ord) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
		double nonsense = 0.0;//we don't need the vectors so just a useless arg
		LAPACKE_dsyevx(ord, 'N', 'I', 'U', n, ATA, n, nonsense, nonsense, n, n, s, &out_m, out_w, out_z, 1, ifail);
		*result = out_w[0];
		delete [] ifail;
		alsm_free<DeviceType::CPU, double>(ATA);
		alsm_free<DeviceType::CPU, double>(out_w);
		printf("the max sigular value square  is %lf\n", static_cast<double>(*result));
	}
	//cuda implementation is not done yet
}
#endif