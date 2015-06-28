#include "solver/l1_min_solver.h"
#include <time.h>
using namespace alsm;
int main()
{
	//m row n column
	clock_t begin, end;
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

	float diffX;
	float diffE;
	float normX;
	float normXG;

	double normB, normX0, normA, normYk;
	normB = 0;
	normX0 = 0;
	normYk = 0;
	for (int i = 0; i < m; i++)//the A is row first stored
	{
		normA = 0;
		for (int j = 0; j < n; j++)//normalize every row's l2 norm to 1
		{
			col_first_A[j*m + i] = (float)1.0 * (rand() - rand()) / 100;
			normA += col_first_A[j*m + i] * col_first_A[j*m + i];
		}
		normA = sqrt(normA);
		for (int j = 0; j < n; j++)
		{
			col_first_A[j*m + i] = (float)(col_first_A[j*m + i] / normA);
		}
	}
	normA = 0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			normA += col_first_A[j*m + i] * col_first_A[j*m + i];
		}
	}
	normX0 = 0;
	for (int j = 0; j < n; j++)
	{
		xG[j] = 0;
	}
	for (int j = 0; j < (int) (n * sparsity); j++)
	{
		xG[(int) (rand()) % n] = (float)1.0 * (rand() - rand());
	}
	for (int j = 0; j < n; j++)
	{
		normX0 += xG[j] * xG[j];
	}
	normX0 = sqrt(normX0);
	for (int j = 0; j < n; j++)
	{
		xG[j] = (float) (xG[j] / normX0);
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
		yk[(int) (rand()*m) % m] = (float)1.0 * (rand() - rand());
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
	printf("normA: %f\n", sqrt(normA));
	printf("normB: %f\n", sqrt(normB));
	printf("normX0: %f\n", sqrt(normX0));
	printf("normYk: %f\n", sqrt(normYk));
	l1_solver<DeviceType::CPU, float> solver(std::array<stream<DeviceType::CPU>, 3>(), m, n, 500, 1);
	solver.init_memory();
	solver.init_problem(MatrixMemOrd::COL, col_first_A, b, output_x, output_e);
	solver.init_parameter(0.01, 0.01, 0.1, 100, 1.5);
	
	begin = clock();
	solver.solve();
	end = clock();
	diffX = 0;
	for (int i = 0; i < n; i++)
	{

		diffX += fabsf(output_x[i] - xG[i]);
	}
	diffE = 0;
	for (int i = 0; i < m; i++)
	{
		diffE += fabsf(output_e[i] - yk[i]);
	}
	printf("the l1 diffx norm is %f, the l1 diffe norm is %f\n", diffX, diffE);
	alsm_free_all();
}