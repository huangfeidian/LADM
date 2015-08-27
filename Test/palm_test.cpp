#include "../ALM/PALM/PALM_cuda.h"
#include "data_generate.h"
#include <string>
using namespace std;
int main()
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
	m = begin_m*5;
	n = begin_n*5;
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
	vector<float> all_tol_int = { 0.00008f, 0.00004f, 0.00002f, 0.00001f, 0.000008f };
	float* lambda = new float[m];
	for (int i = 0; i < all_tol_int.size(); i++)
	{
		string file_name = "palm_tol";
		file_name += to_string(i) + ".csv";
		FILE* palm_log = fopen(file_name.c_str(), "w");
		L1Solver_e1x1 PLAM_solver(m, n);
		PLAM_solver.set_A(A);
		PLAM_solver.set_logFile(palm_log);
		PLAM_solver.solve(b, output_x, output_e, 0.001, all_tol_int[i], 5000, 50, 8, xG, 0.001);
	}
	
	delete [] output;
	delete [] result_b;
	delete [] lambda;

	delete [] A;
	delete [] b;
	delete [] xG;
	
}