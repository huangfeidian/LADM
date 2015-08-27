#ifndef __H_DATA_GENERATE_H__
#define __H_DATA_GENERATE_H__
#include <random>
#include <fstream>
#include "../blas/level_1.h"
#include "../blas/level_2.h"
using namespace alsm;
template <typename T>
void load_data(T* A, T* b, int m, int n, std::string A_file, std::string b_file)
{
	std::ifstream A_stream(A_file);
	std::ifstream B_stream(b_file);
	for (int i = 0; i < m; i++)
	{
		B_stream >> *(b + i);
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			A_stream >> *(A + i*m + j);
		}
	}
}
template<typename T>
T normal_matrix(T* A, int m, int n)//A is col first
{
	T normA = 0;
	for (int i = 0; i < n; i++)//the A is col first stored
	{
		normA = 0;
		for (int j = 0; j < m; j++)//normalize every row's l2 norm to 1
		{
			normA += A[i*m + j] * A[i*m + j];
		}
		normA = sqrt(normA);
		for (int j = 0; j < m; j++)
		{
			A[i*m + j] = (T) (A[i*m + j] / normA);
		}
	}
	normA = 0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			normA += A[j*m + i] * A[j*m + i];
		}
	}
	return normA;
}
template<typename T>
void generate_random(T* in_vec, int length, T scarcity = 1)
{
	std::uniform_real_distribution<T> uniform_dist(-10, 10);
	std::uniform_int_distribution<int> spacity_dist(0);
	std::mt19937 seed(109);
	if (scarcity == 1)
	{
		for (int i = 0; i < length; i++)
		{
			in_vec[i] = uniform_dist(seed);
		}
	}
	else
	{
		for (int i = 0; i < length*scarcity; i++)
		{
			in_vec[spacity_dist(seed) % length] = uniform_dist(seed);
		}
	}

}
template<typename T>
void normalize_vector(T* in_vec, int length)
{
	T norm = 0;
	for (int i = 0; i < length; i++)
	{
		norm += in_vec[i] * in_vec[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < length; i++)
	{
		in_vec[i] = in_vec[i] / norm;
	}
}
template<typename T>
T generate_test_data(T* A, T* x, T* e, T* b, int m, int n, T scarcity)
{
	generate_random(A, m*n);
	normal_matrix(A, m, n);
	generate_random(x, n, scarcity);
	normalize_vector(x, n);
	generate_random(e, m, scarcity);
	normalize_vector(e, m);
	stream<DeviceType::CPU> main_cpu_stream;
	copy<DeviceType::CPU, T>(main_cpu_stream, m, e, b);
	gemv<DeviceType::CPU, T>(main_cpu_stream, MatrixTrans::NORMAL, MatrixMemOrd::COL, m, n, 1, A, m, x, 1, b);
	T e_opt = 0;
	T x_opt = 0;
	asum<DeviceType::CPU, T>(main_cpu_stream, m, e, &e_opt);
	asum<DeviceType::CPU, T>(main_cpu_stream, n, x, &x_opt);
	return e_opt + x_opt;
}
#endif