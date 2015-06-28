#ifndef __H_proximal_H__
#define __H_proximal_H__
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "proximal_eval.h"
#include "proximal_subgradient.h"
#include "../util/stream.h"
namespace alsm
{
	template <DeviceType D, typename T>
	void BatchProxEval(const stream<D>& stream,  const FunctionObj<T>&  f_obj, int size, T rho, const T *x_in,
		T *x_out);


	template <DeviceType D, typename T>
	void BatchFuncEval(const stream<D>& stream,  const FunctionObj<T>&  f_obj, int size, const T *x_in,T* result);


	template <DeviceType D, typename T>
	void BatchSubgradEval(const stream<D>& stream, const FunctionObj<T>&  f_obj, int size, const T *x_in, const T *v_in,
		T *v_out);


	template <>
	__DEVICE__ void BatchProxEval<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, const FunctionObj<float>&  f_obj, int size, float rho, const float *x_in,
		float *x_out)
	{
#if ALSM_USE_OMP
#pragma omp parallel for
#endif
		for (unsigned int i = 0; i < size; ++i)
		{
			x_out[i] = ProxEval<float>(f_obj, x_in[i], rho);
		}
	}


	template <>
	__DEVICE__ void BatchProxEval<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream, const FunctionObj<double>&  f_obj, int size, double rho, const double *x_in,
		double *x_out)
	{
#if ALSM_USE_OMP
#pragma omp parallel for
#endif
		//printf(" the rho is %lf\n", rho);
		//printf("a:%lf'\tb:%lf\tc:%lf\td:%lf\te:%lf\n", f_obj.a, f_obj.b, f_obj.c, f_obj.d, f_obj.e);
		for (unsigned int i = 0; i < size; ++i)
		{
			x_out[i] = ProxEval<double>(f_obj, x_in[i], rho);
			//printf("%lf,", x_out[i]);
		}
		//printf("\n");
	}
	// Returns evalution of Sum_i Func{f_obj[i]}(x_in[i]).
	//
	// @param f_obj Vector of function objects.
	// @param x_in Array to which function will be applied.
	// @param x_out Array to which result will be written.
	// @returns Evaluation of sum of functions.

	template <>
	__DEVICE__ void  BatchFuncEval<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, const FunctionObj<float>& f_obj, int size, const float* x_in, float* result)
	{
		float sum = 0;
#if ALSM_USE_OMP
#pragma omp parallel for reduction(+:sum)
#endif
		for (unsigned int i = 0; i < size; ++i)
		{
			sum += FuncEval<float>(f_obj, x_in[i]);
		}
		*result = sum;
	}
	template <>
	__DEVICE__ void  BatchFuncEval<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream, const FunctionObj<double>&  f_obj, int size, const double* x_in, double* result)
	{
		double sum = 0;
#if ALSM_USE_OMP
#pragma omp parallel for reduction(+:sum)
#endif
		for (unsigned int i = 0; i < size; ++i)
		{
			sum += FuncEval<double>(f_obj, x_in[i]);
		}
		*result = sum;
	}
	// Projection onto the subgradient at x_in
	//   ProjSubgrad{f_obj[i]}(x_in[i], v_in[i]) -> x_out[i].
	//
	// @param f_obj Vector of function objects.
	// @param x_in Array of points at which subgradient should be evaluated.
	// @param v_in Array of points that should be projected onto the subgradient.
	// @param v_out Array to which result will be written.



	template <>
	__DEVICE__ void BatchSubgradEval<DeviceType::CPU, float>(const stream<DeviceType::CPU>& stream, const FunctionObj<float>& f_obj, int size, const float *x_in,
		const float *v_in, float *v_out)
	{
#if ALSM_USE_OMP
#pragma omp parallel for
#endif
		for (unsigned int i = 0; i < size; ++i)
		{
			v_out[i] = SubgradEval<float>(f_obj, v_in[i], x_in[i]);
		}
	}
	template <>
	__DEVICE__ void BatchSubgradEval<DeviceType::CPU, double>(const stream<DeviceType::CPU>& stream, const FunctionObj<double>& f_obj, int size, const double *x_in,
		const double *v_in, double *v_out)
	{
#if ALSM_USE_OMP
#pragma omp parallel for
#endif
		for (unsigned int i = 0; i < size; ++i)
		{
			v_out[i] = SubgradEval<double>(f_obj, v_in[i], x_in[i]);
		}
	}
#if ALSM_USE_CUDA
	extern void thrust_func_eval_s(const stream<DeviceType::GPU>& stream, const FunctionObj<float>  &f_obj, int size, const float *x_in,float* result);
	template <>
	__DEVICE__ void BatchFuncEval<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, const FunctionObj<float>  &f_obj, int size, const float *x_in,float* result)
	{
		thrust_func_eval_s(stream, f_obj, size, x_in,result);
	}

	extern void thrust_func_eval_d(const stream<DeviceType::GPU>& stream, const FunctionObj<double>  &f_obj, int size, const double *x_in,double* result);
	template <>
	__DEVICE__ void BatchFuncEval<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, const FunctionObj<double>  &f_obj, int size, const double *x_in,double* result)
	{
		thrust_func_eval_d(stream, f_obj, size, x_in,result);
	}



	extern void thrust_func_prox_s(const stream<DeviceType::GPU>& stream, const FunctionObj<float> &f_obj, int size, float rho,
		const float *x_in, float *x_out);
	template <>
	__DEVICE__ void BatchProxEval<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, const FunctionObj<float> &f_obj, int size, float rho,
		const float *x_in, float *x_out)
	{
		thrust_func_prox_s(stream, f_obj, size, rho,x_in, x_out);
	}

	extern void thrust_func_prox_d(const stream<DeviceType::GPU>& stream, const FunctionObj<double> &f_obj, int size, double rho,
		const double *x_in, double *x_out);
	template <>
	__DEVICE__ void BatchProxEval<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, const FunctionObj<double> &f_obj, int size, double rho,
		const double *x_in, double *x_out)
	{
		thrust_func_prox_d(stream, f_obj, size, rho, x_in, x_out);

	}
	

	
	extern void thrust_func_subgrad_s(const stream<DeviceType::GPU>& stream, const FunctionObj<float> &f_obj, int size,
		const float *x_in, const float *v_in, float *v_out);
	template <>
	__DEVICE__ void BatchSubgradEval<DeviceType::GPU, float>(const stream<DeviceType::GPU>& stream, const FunctionObj<float> &f_obj, int size,
		const float *x_in, const float *v_in, float *v_out)
	{
		thrust_func_subgrad_s(stream, f_obj, size, x_in, v_in, v_out);
	}
	extern void thrust_func_subgrad_d(const stream<DeviceType::GPU>& stream, const FunctionObj<double> &f_obj, int size,
		const double *x_in, const double *v_in, double *v_out);
	template <>
	__DEVICE__ void BatchSubgradEval<DeviceType::GPU, double>(const stream<DeviceType::GPU>& stream, const FunctionObj<double> &f_obj, int size,
		const double *x_in, const double *v_in, double *v_out)
	{
		thrust_func_subgrad_d(stream, f_obj, size, x_in, v_in, v_out);
	}
#endif  //
}
#endif
