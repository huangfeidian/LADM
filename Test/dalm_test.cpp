#include "../ALM/DALM/DALM_fast.h"
#include "data_generate.h"
#include <string>
using namespace std;
void test_tol_int()
{
	int begin_m = 320*8;
	int begin_n = 1024*8;
	float* A;
	float* b;
	float* x;
	float* e;
	float* xG;
	float opt_G;

	int m, n;
	m = begin_m * 1;
	n = begin_n * 1;
	A = new float[m*(n + m)];
	b = new float[m];

	xG = new float[m + n];
	x = xG;
	e = xG + n;
	memset(A, 0, m*n*sizeof(float));
	memset(b, 0, m*sizeof(float));
	memset(e, 0, m*sizeof(float));
	memset(x, 0, n*sizeof(float));
	opt_G = generate_test_data<float>(A, x, e, b, m, n, 0.1);
	cout << "opt:" << opt_G << endl;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (i != j)
			{
				A[m*n + i*m + j] = 0;
			}
			else
			{
				A[m*n + i*m + j] = 1;
			}
		}
	}
	float* output = new float[m + n];
	float* output_x = output;
	float* output_e = output + n;
	float* result_b = new float[m];
	vector<float> all_tol_int = {0.000001f };
	//0.00008f,0.00004f ,0.00002f ,0.000005f
	float* lambda = new float[m];
	for (int i = 0; i < all_tol_int.size(); i++)
	{
		string file_name = "dalm_tol";
		file_name += to_string(i) + ".csv";
		FILE* dalm_log = fopen(file_name.c_str(), "w");
		DALM_solver dalm_solver(m, n + m, 8, 0.001,all_tol_int[i],0.001,10000);
		dalm_solver.allocate_memory();
		dalm_solver.set_log_file(dalm_log);
		dalm_solver.solve(output, b, A, xG);
		dalm_solver.free_memory();
	}

	delete [] output;
	delete [] result_b;
	delete [] lambda;

	delete [] A;
	delete [] b;
	delete [] xG;
}
void test_input_dimension()
{
	int begin_m = 320;
	int begin_n = 1024;
	float* A;
	float* b;
	float* x;
	float* e;
	float* xG;
	float opt_G;

	int m, n;
	for (int k = 1; k < 10; k++)
	{
		m = begin_m * k;
		n = begin_n * k;
		A = new float[m*(n + m)];
		b = new float[m];

		xG = new float[m + n];
		x = xG;
		e = xG + n;
		memset(A, 0, m*n*sizeof(float));
		memset(b, 0, m*sizeof(float));
		memset(e, 0, m*sizeof(float));
		memset(x, 0, n*sizeof(float));
		opt_G = generate_test_data<float>(A, x, e, b, m, n, 0.1);
		cout << "opt:" << opt_G << endl;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
			{
				if (i != j)
				{
					A[m*n + i*m + j] = 0;
				}
				else
				{
					A[m*n + i*m + j] = 1;
				}
			}
		}
		float* output = new float[m + n];
		float* output_x = output;
		float* output_e = output + n;
		float* result_b = new float[m];
		float* lambda = new float[m];

		string file_name = "dalm_tol";
		file_name += to_string(k) + ".csv";
		FILE* dalm_log = fopen(file_name.c_str(), "w");
		DALM_solver dalm_solver(m, n + m, 8, 0.001, 0.000001);
		dalm_solver.allocate_memory();
		dalm_solver.set_log_file(dalm_log);
		dalm_solver.solve(output, b, A, xG);
		dalm_solver.free_memory();
		delete [] output;
		delete [] result_b;
		delete [] lambda;

		delete [] A;
		delete [] b;
		delete [] xG;
	}


}
int main()
{
	test_tol_int();

}