#include "../solver/multi_seq.h"
#include "data_generate.h"
using namespace std;
#define EPS1 0.001
#define EPS2 0.15
#define EPS3 0.005
void test_input()
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
		for (int i = 0; i <m; i++)
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

		string file_name = "LADM_input_";
		file_name += std::to_string(k) + ".csv";
		FILE* ladm_log = fopen(file_name.c_str(), "w");
		memset(output, 0, sizeof(float)*(m + n));
		memset(lambda, 0, sizeof(float)*m);
		memset(result_b, 0, sizeof(float)*m);
		std::vector<stream<DeviceType::GPU>> multi_gpu_streams = stream<DeviceType::GPU>::create_streams(1 + 2, false);
		multi_seq<DeviceType::GPU, float> multi_seq_gpu_solver1(multi_gpu_streams[1 + 1], 1 + 1, m, 5000, 10);
		multi_seq_gpu_solver1.init_memory();
		multi_seq_gpu_solver1.init_server(multi_gpu_streams[1 + 1], b, lambda, StopCriteria::file_log, opt_G);
		for (int i = 0; i <1; i++)
		{
			multi_seq_gpu_solver1.add_client(multi_gpu_streams[i], n / 1, FunctionObj<float>(UnaryFunc::Abs), A + m*(n / 1)*i, false,
				MatrixMemOrd::COL, output_x + (n / 1)*i, 0, xG + (n / 1)*i);
		}
		multi_seq_gpu_solver1.set_log_file(ladm_log);
		multi_seq_gpu_solver1.add_client(multi_gpu_streams[1], m, FunctionObj<float>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e, 0, xG + n);
		multi_seq_gpu_solver1.init_parameter(EPS1, 0.15, 1, 10000, 1.1, EPS3);
		//begin = std::chrono::high_resolution_clock::now();
		multi_seq_gpu_solver1.solve();
		//end = std::chrono::high_resolution_clock::now();
		//elapsed = end - begin;
		alsm_free_all();
		for (auto j : multi_gpu_streams)
		{
			j.destory();
		}
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
	test_input();

}