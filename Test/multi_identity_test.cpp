#include "../solver/multi_para.h"
#include "../solver/multi_seq.h"
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <string>

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
void generate_random(float* in_vec, int length)
{
	std::random_device rd;
	std::uniform_real_distribution<float> uniform_dist(0, 1);
	std::mt19937 seed(rd());
	for (int i = 0; i < length; i++)
	{
		in_vec[i] = uniform_dist(seed);
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
int main()
{
	//m row n column

	int alg = 3;
	int m = 1024 * 2;
	int n = 1024*2;
	int maxIterInner = 50;
	int maxIterOuter = 5000;
	float sparsity = 0.1;
	float tol = 0.01;
	float tol_int = 0.01;
	float* b = new float[n];
	float normB = 0;
	float original_opt = 0;
	float* lambda = new float[n];
	float* matrix_A[4];
	float* output_x[4];
	memset(lambda, 0, sizeof(float)*n);
	for (int i = 0; i < 4;i++)
	{
		output_x[i] = new float[n];
		memset(output_x[i], 0, sizeof(float)*n);
		matrix_A[i] = new float[m*n];
		memset(matrix_A[i], 0, sizeof(float)*m*n);
		for (int j = 0; j < m; j++)
		{
			matrix_A[i][j*m + j] = 1;
		}
	}

	stream<DeviceType::CPU> main_cpu_stream;
	generate_random(b, n);
	normalize_vector(b, n);
	for (int i = 0; i < n; i++)
	{
		normB += b[i] * b[i];
		original_opt += b[i];
	}
	printf("normB: %f\n", sqrt(normB));

	printf("original_opt: %f\n", original_opt);

	std::array<stream<DeviceType::CPU>, 5> streams{};
	//for (int i = 0; i < 5; i++)
	//{
	//	cudaStream_t temp_stream;
	//	cudaStreamCreate(&temp_stream);
	//	streams[i] = stream<DeviceType::GPU>(temp_stream);
	//}
	multi_seq<DeviceType::CPU, float> solver(streams[4], 4, n, 1000, 10);
	solver.init_memory();
	solver.init_server(streams[4], b, lambda);
	for (int i = 0; i < 4; i++)
	{
		solver.add_client(streams[i], n, FunctionObj<float>(UnaryFunc::Abs), matrix_A[i],true, MatrixMemOrd::COL, output_x[i]);
	}
	solver.init_parameter(0.01, 0.01, 10, 1000, 1.1);
	auto begin = std::chrono::high_resolution_clock::now();
	solver.solve();
	auto end = std::chrono::high_resolution_clock::now();
	float opt = 0;
	float abs_x = 0;
	for (int i = 0; i < 4; i++)
	{
		float temp_result;
		asum<DeviceType::CPU, float>(main_cpu_stream, n, output_x[i],&temp_result);
		opt += temp_result;
	}
	std::chrono::duration<float, std::milli> elapsed = end - begin;
	std::cout << "the opt is " << opt << " error is " << original_opt-opt << " time is " << elapsed.count() << std::endl;
	//printf("the l1 diffx norm is %f, the l1 diffe norm is %f the opt is %f time is %dms\n", diffX, diffE,opt,end-begin);
	alsm_free_all();
}