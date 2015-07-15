#include "solver/para_l1.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

using namespace alsm;
void load_data(float* A, float* b,int m,int n, std::string A_file, std::string b_file)
{
	std::ifstream A_stream(A_file);
	std::ifstream B_stream(b_file);
	for (int i = 0; i < m; i++)
	{
		B_stream >> *(b+i);
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
int main()
{
	//m row n column
	
	int alg = 3;
	int m = 1024;
	int n = 32 * 10;
	int maxIterInner = 50;
	int maxIterOuter = 5000;
	float sparsity = 0.1;
	float tol = 0.01;
	float tol_int = 0.01;
	int iter;
	int seed = 100;

	srand(seed);

	float *b = new float[m];
	float *yk = new float[m];
	float *col_first_A = new float[m*n];
	float *x = new float[n];
	float *xG = new float[n];
	float *e = new float[m];
	float *output_e=new float[m];
	float* output_x = new float[n];
	float original_opt=0;
	float diffX;
	float diffE;
	float normX;
	float normXG;

	float normB, normX0, normA, normYk;
	normB = 0;
	normX0 = 0;
	normYk = 0;
	for (int i = 0; i < m; i++)//the A is row first stored
	{
		for (int j = 0; j < n; j++)
		{
			col_first_A[j*m + i] = (float)1.0 * (rand() - rand()) / 100;
			
		}
	}
	normA = normal_matrix(col_first_A, m, n);
	normX0 = 0;
	for (int j = 0; j < n; j++)
	{
		xG[j] = 0;
	}
	for (int j = 0; j < (int) (n * sparsity); j++)
	{
		xG[(int) (rand()) % n] = fabsf((float)1.0 * (rand() - rand()));
	}
	for (int j = 0; j < n; j++)
	{
		normX0 += xG[j] * xG[j];
	}
	normX0 = sqrt(normX0);
	for (int j = 0; j < n; j++)
	{
		xG[j] = (float) (xG[j] / normX0);
		original_opt += xG[j];
	}
	normX0 = 0;
	for (int j = 0; j < n; j++)
	{
		normX0 += xG[j] * xG[j];
	}
	normYk = 0;
	for (int j = 0; j < m; j++)
	{
		yk[j] = 0;
	}
	for (int j = 0; j < (int) (m * sparsity); j++)
	{
		yk[(int) (rand()) % m] = fabsf((float)1.0 * (rand() - rand()));
	}
	for (int j = 0; j < m; j++)
	{
		normYk += yk[j] * yk[j];
	}
	normYk = sqrt(normYk);
	for (int j = 0; j < m; j++)
	{
		e[j] = 0;
		yk[j] = (float) (yk[j] / normYk);
		original_opt += yk[j];
	}
	normYk = 0;
	for (int j = 0; j < m; j++)
	{
		normYk += yk[j] * yk[j];
	}
	for (int i = 0; i < m; i++)
	{
		b[i] = yk[i];
	}
	stream<DeviceType::CPU> main_cpu_stream;
	float one = 1.0;
	//b=yk+A*xG
	gemv<DeviceType::CPU,float>(main_cpu_stream,MatrixTrans::NORMAL,MatrixMemOrd::COL, m, n,&one, col_first_A, m, xG, &one, b);
	for (int i = 0; i < m; i++)
	{
		normB += b[i] * b[i];
	}
	//load_data(col_first_A, b, m, n, "A_matrix.txt", "b_vector.txt");
	printf("normA: %f\n", sqrt(normA));
	printf("normB: %f\n", sqrt(normB));
	printf("normX0: %f\n", sqrt(normX0));
	printf("normYk: %f\n", sqrt(normYk));
	printf("original_opt: %f\n", original_opt);
	//FILE * A_out = fopen("A_matrix.txt", "wb");
	//FILE* b_out = fopen("b_vector.txt", "wb");
	//for (int i = 0; i < n; i++)
	//{
	//	for (int j = 0; j < m; j++)
	//	{
	//		fprintf(A_out, "%f ", col_first_A[i*m + j]);
	//	}
	//	fprintf(A_out, "\n");
	//	
	//}
	//for (int i = 0; i < m; i++)
	//{
	//	fprintf(b_out, "%f ", b[i]);
	//}
	std::array<stream<DeviceType::CPU>, 3> streams{};
	/*for (int i = 0; i < 3; i++)
	{
		cudaStream_t temp_stream;
		cudaStreamCreate(&temp_stream);
		streams[i] = stream<DeviceType::GPU>(temp_stream);
	}*/
	para_l1<DeviceType::CPU, float> solver(streams, m, n, 500, 10);
	solver.init_memory();
	solver.init_problem(MatrixMemOrd::COL, col_first_A, b, output_x, output_e);
	solver.init_parameter(0.01, 0.01, 4, 1000, 1.1);
	
	
	
	auto begin = std::chrono::high_resolution_clock::now();
	solver.solve();
	auto end = std::chrono::high_resolution_clock::now();
	diffX = 0;
	float opt = 0;
	for (int i = 0; i < n; i++)
	{

		opt += output_x[i];
	}
	diffE = 0;
	for (int i = 0; i < m; i++)
	{
		opt += fabsf(output_e[i]);
	}
	gemv<DeviceType::CPU, float>(main_cpu_stream, MatrixTrans::NORMAL, MatrixMemOrd::COL, m, n, &one, col_first_A, m, output_x, &one, output_e);
	float neg_one = -1;
	axpy<DeviceType::CPU, float>(main_cpu_stream, m, &neg_one, b, output_e);
	float error = 0;
	nrm2<DeviceType::CPU, float>(main_cpu_stream, m,output_e, &error);
	std::chrono::duration<float, std::milli> elapsed = end - begin;
	std::cout << "the opt is " << opt << " error is "<<error<<" time is " << elapsed.count() << std::endl;
	//printf("the l1 diffx norm is %f, the l1 diffe norm is %f the opt is %f time is %dms\n", diffX, diffE,opt,end-begin);
	alsm_free_all();
}