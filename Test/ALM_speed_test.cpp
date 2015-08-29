#include "data_generate.h"
#include "../solver/multi_para.h"
#include "../solver/multi_seq.h"
#include <fstream>
#include <iostream>
using namespace std;
using namespace alsm;
void test(ofstream& output_file, float* A, float*b, float*in_output_x, float* in_output_e, float* in_lambda, int m, int n)
{

	vector<int> blocks{ 1, 2, 4, 5, 8 };
	auto begin = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::milli> elapsed;
	float* output_x = new float[n];
	float* output_e = new float[m];
	float* lambda = new float[m];
	for (auto block_size : blocks)
	{

#if ALSM_USE_GPU
		//Begin GPU
		std::vector<stream<DeviceType::GPU>> gpu_streams = stream<DeviceType::GPU>::create_streams(block_size + 2, false);
		multi_seq<DeviceType::GPU, float> seq_gpu_solver(gpu_streams[block_size + 1], block_size + 1, m, 500, 10);
		seq_gpu_solver.init_memory();
		seq_gpu_solver.init_server(gpu_streams[block_size + 1], b, lambda,StopCriteria::kkt_dual_tol);
		for (int i = 0; i <block_size; i++)
		{
			seq_gpu_solver.add_client(gpu_streams[i], n / block_size, FunctionObj<float>(UnaryFunc::Abs), A + m*(n / block_size)*i, false, MatrixMemOrd::COL, output_x + (n / block_size)*i);
		}
		seq_gpu_solver.add_client(gpu_streams[block_size], m, FunctionObj<float>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e);
		seq_gpu_solver.init_parameter(0.001, 0.15, 1, 10000, 1.1,0.0001);
		begin = std::chrono::high_resolution_clock::now();
		seq_gpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		output_file << "GPU SEQ" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << seq_gpu_solver.lambda_server.current_eps1 << "," << seq_gpu_solver.lambda_server.total_opt_value << endl;
		alsm_free_all();
		//multi para
		multi_para<DeviceType::GPU, float> para_gpu_solver(gpu_streams[block_size + 1], block_size + 1, m, 500, 10);
		para_gpu_solver.init_memory();
		para_gpu_solver.init_server(gpu_streams[block_size + 1], b, lambda,StopCriteria::kkt_dual_tol);
		for (int i = 0; i <block_size; i++)
		{
			para_gpu_solver.add_client(gpu_streams[i], n / block_size, FunctionObj<float>(UnaryFunc::Abs), A + m*(n / block_size)*i, false, MatrixMemOrd::COL, output_x + (n / block_size)*i);
		}
		para_gpu_solver.add_client(gpu_streams[block_size], m, FunctionObj<float>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e);
		para_gpu_solver.init_parameter(0.001, 0.15, 1, 10000, 1.1, 0.0001);
		begin = std::chrono::high_resolution_clock::now();
		para_gpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		output_file << "GPU PARA" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << para_gpu_solver.lambda_server.current_eps1 << "," << para_gpu_solver.lambda_server.total_opt_value << endl;
		alsm_free_all();
		for (auto i : gpu_streams)
		{
			i.destory();
		}
		//begin multi gpu
		std::vector<stream<DeviceType::GPU>> multi_gpu_streams = stream<DeviceType::GPU>::create_streams(block_size + 2, true);
		multi_seq<DeviceType::GPU, float> multi_seq_gpu_solver(multi_gpu_streams[block_size + 1], block_size + 1, m, 500, 10);
		multi_seq_gpu_solver.init_memory();
		multi_seq_gpu_solver.init_server(multi_gpu_streams[block_size + 1], b, lambda);
		for (int i = 0; i <block_size; i++)
		{
			multi_seq_gpu_solver.add_client(multi_gpu_streams[i], n / block_size, FunctionObj<float>(UnaryFunc::Abs), A + m*(n / block_size)*i, false, MatrixMemOrd::COL, output_x + (n / block_size)*i);
		}
		multi_seq_gpu_solver.add_client(multi_gpu_streams[block_size], m, FunctionObj<float>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e);
		multi_seq_gpu_solver.init_parameter(0.01, 0.01, 1, 1000, 1.1);
		begin = std::chrono::high_resolution_clock::now();
		multi_seq_gpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		output_file << "MULTI GPU SEQ" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << multi_seq_gpu_solver.lambda_server.current_eps1 << "," << multi_seq_gpu_solver.lambda_server.total_opt_value << endl;
		alsm_free_all();
		multi_para<DeviceType::GPU, float> multi_para_gpu_solver(multi_gpu_streams[block_size + 1], block_size + 1, m, 500, 10);
		multi_para_gpu_solver.init_memory();
		multi_para_gpu_solver.init_server(multi_gpu_streams[block_size + 1], b, lambda);
		for (int i = 0; i <block_size; i++)
		{
			multi_para_gpu_solver.add_client(multi_gpu_streams[i], n / block_size, FunctionObj<float>(UnaryFunc::Abs), A + m*(n / block_size)*i, false, MatrixMemOrd::COL, output_x + (n / block_size)*i);
		}
		multi_para_gpu_solver.add_client(multi_gpu_streams[block_size], m, FunctionObj<float>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e);
		multi_para_gpu_solver.init_parameter(0.01, 0.01, 1, 1000, 1.1);
		begin = std::chrono::high_resolution_clock::now();
		multi_para_gpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		output_file << "MULTI GPU PARA" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << multi_para_gpu_solver.lambda_server.current_eps1 << "," << multi_para_gpu_solver.lambda_server.total_opt_value << endl;
		alsm_free_all();
		for (auto i : multi_gpu_streams)
		{
			i.destory();
		}
#endif

	}
	delete [] output_e;
	delete [] output_x;
	delete [] lambda;
}
int main()
{
	int begin_m = 1024;
	int begin_n = 320;
	float* A;
	float* b;
	float* lambda;
	float* x;
	float* e;
	ofstream output("speedtest.csv");
	output << "type,block_size,m,n,time,eps,opt" << endl;
	for (int i = 2; i <= 20; i++)
	{
		int m, n;
		m = begin_m*i;
		n = begin_n*i;
		A = new float[m*n];
		b = new float[m];
		lambda = new float[m];
		x = new float[n];
		e = new float[m];
		memset(A, 0, m*n*sizeof(float));
		memset(b, 0, m*sizeof(float));
		memset(e, 0, m*sizeof(float));
		memset(lambda, 0, m*sizeof(float));
		memset(x, 0, n*sizeof(float));
		generate_test_data(A, x, e, b, m, n, 0.1f);
		test(output, A, b, x, e, lambda, m, n);
		delete [] A;
		delete [] b;
		delete [] lambda;
		delete [] x;
		delete [] e;
	}
}