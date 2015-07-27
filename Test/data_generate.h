#ifndef __H_DATA_GENERATE_H__
#define __H_DATA_GENERATE_H__
#include <random>
#include <fstream>
#include "../blas/level_1.h"
#include "../blas/level_2.h"
using namespace alsm;
void load_data(float* A, float* b, int m, int n, std::string A_file, std::string b_file)
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
float normal_matrix(float* A, int m, int n)//A is col first
{
	float normA = 0;
	for (int i = 0; i < m; i++)//the A is col first stored
	{
		normA = 0;
		for (int j = 0; j < n; j++)//normalize every row's l2 norm to 1
		{
			normA += A[j*m + i] * A[j*m + i];
		}
		normA = sqrt(normA);
		for (int j = 0; j < n; j++)
		{
			A[j*m + i] = (float) (A[j*m + i] / normA);
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
void generate_random(float* in_vec, int length, float spacity = 1)
{
	std::random_device rd;
	std::uniform_real_distribution<float> uniform_dist(0, 1);
	std::uniform_int_distribution<int> spacity_dist(0);
	std::mt19937 seed(rd());
	if (spacity == 1)
	{
		for (int i = 0; i < length; i++)
		{
			in_vec[i] = uniform_dist(seed);
		}
	}
	else
	{
		for (int i = 0; i < length*spacity; i++)
		{
			in_vec[spacity_dist(seed) % length] = uniform_dist(seed);
		}
	}

}
void normalize_vector(float* in_vec, int length)
{
	float norm = 0;
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
float generate_test_data(float* A, float* x, float* e, float* b, int m, int n, float spacity)
{
	generate_random(A, m*n);
	normal_matrix(A, m, n);
	generate_random(x, n, spacity);
	normalize_vector(x, n);
	generate_random(e, n, spacity);
	normalize_vector(e, n);
	stream<DeviceType::CPU> main_cpu_stream;
	copy<DeviceType::CPU, float>(main_cpu_stream, m, e, b);
	gemv<DeviceType::CPU, float>(main_cpu_stream, MatrixTrans::NORMAL, MatrixMemOrd::COL, m, n, 1, A, m, x, 1, b);
	float e_opt = 0;
	float x_opt = 0;
	asum<DeviceType::CPU, float>(main_cpu_stream, m, e, &e_opt);
	asum<DeviceType::CPU, float>(main_cpu_stream, n, x, &x_opt);
	return e_opt + x_opt;
}
#endif