#include "../ALM/DALM/DALM_fast.h"
#include "data_generate.h"
#include <string>
using namespace std;
void test_tol_int()
{
	int begin_m = 320*8;
	int begin_n = 1024*8;
	double* A;
	double* b;
	double* x;
	double* e;
	double* xG;
	double opt_G;

	int m, n;
	m = begin_m * 1;
	n = begin_n * 1;
	A = new double[m*(n + m)];
	b = new double[m];

	xG = new double[m + n];
	x = xG;
	e = xG + n;
	memset(A, 0, m*n*sizeof(double));
	memset(b, 0, m*sizeof(double));
	memset(e, 0, m*sizeof(double));
	memset(x, 0, n*sizeof(double));
	opt_G = generate_test_data<double>(A, x, e, b, m, n, 0.1);
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
	double* output = new double[m + n];
	double* output_x = output;
	double* output_e = output + n;
	double* result_b = new double[m];
	vector<double> all_tol_int = {0.000001 };
	//0.00008f,0.00004f ,0.00002f ,0.000005f
	double* lambda = new double[m];
	for (int i = 0; i < all_tol_int.size(); i++)
	{
		string file_name = "dalm_tol";
		file_name += to_string(i) + ".csv";
		FILE* dalm_log = fopen(file_name.c_str(), "w");
		dalmDsolver dalm_solver(m, n + m, 8, 0.001,all_tol_int[i],0.001,10000);
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
	double* A;
	double* b;
	double* x;
	double* e;
	double* xG;
	double opt_G;

	int m, n;
	for (int k = 1; k < 10; k++)
	{
		m = begin_m * k;
		n = begin_n * k;
		A = new double[m*(n + m)];
		b = new double[m];

		xG = new double[m + n];
		x = xG;
		e = xG + n;
		memset(A, 0, m*n*sizeof(double));
		memset(b, 0, m*sizeof(double));
		memset(e, 0, m*sizeof(double));
		memset(x, 0, n*sizeof(double));
		opt_G = generate_test_data<double>(A, x, e, b, m, n, 0.1);
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
		double* output = new double[m + n];
		double* output_x = output;
		double* output_e = output + n;
		double* result_b = new double[m];
		double* lambda = new double[m];

		string file_name = "dalm_tol";
		file_name += to_string(k) + ".csv";
		FILE* dalm_log = fopen(file_name.c_str(), "w");
		dalmDsolver dalm_solver(m, n + m, 8, 0.001, 0.000001);
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