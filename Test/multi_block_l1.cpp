#include "../solver/multi_para.h"
#include "../solver/multi_seq.h"
#include "data_generate.h"
#include <chrono>

#include <iostream>

#define BLOCKS 1
using namespace alsm;
int main()
{
	//m row n column
	int m = 32 * 10*5  ;
	int n = 1024*5 ;
	int maxIterInner = 50;
	int maxIterOuter = 5000;
	double sparsity = 0.1f;
	double tol = 0.01f;
	double tol_int = 0.01f;
	stream<DeviceType::CPU> main_cpu_stream;
	double*b = alsm_malloc<DeviceType::CPU, double>(main_cpu_stream,m);
	double* A = alsm_malloc<DeviceType::CPU, double>(main_cpu_stream, m*n);
	double* x = alsm_malloc<DeviceType::CPU, double>(main_cpu_stream, n);
	double* e = alsm_malloc<DeviceType::CPU, double>(main_cpu_stream, m);
	double *output_e = alsm_malloc<DeviceType::CPU, double>(main_cpu_stream, m);
	double* output_x = alsm_malloc<DeviceType::CPU, double>(main_cpu_stream, n);;
	double* lambda = alsm_malloc<DeviceType::CPU, double>(main_cpu_stream, m);
	double original_opt = generate_test_data(A,x,e,b,m,n,sparsity);

	printf("original_opt: %f\n", original_opt);
	
	//std::array<stream<DeviceType::CPU>, BLOCKS+2> streams{};
	std::vector<stream<DeviceType::CPU>> streams = stream<DeviceType::CPU>::create_streams(BLOCKS + 2, false);
	multi_seq<DeviceType::CPU, double> solver(streams[BLOCKS+1], BLOCKS+1, m, 2000, 10);
	solver.init_memory();
	solver.init_server(streams[BLOCKS+1], b, lambda,StopCriteria::objective_value,original_opt);
	for (int i = 0; i <BLOCKS; i++)
	{
		solver.add_client(streams[i], n / BLOCKS, FunctionObj<double>(UnaryFunc::Abs), A + m*(n / BLOCKS)*i, false, MatrixMemOrd::COL, output_x + (n / BLOCKS)*i, 0,x + (n / BLOCKS)*i);
	}
	solver.add_client(streams[BLOCKS], m, FunctionObj<double>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e,0,e);
	solver.init_parameter(0.01f, 0.01f, m*0.001, 100, 1.1f,0.005f);
	auto begin = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	begin = std::chrono::high_resolution_clock::now();
	solver.solve();
	end=std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - begin;
	std::cout << "the opt is " << solver.lambda_server.total_opt_value << " error percentage is " << solver.lambda_server.current_eps1 << " time is " << elapsed.count() << std::endl;
	//printf("the l1 diffx norm is %f, the l1 diffe norm is %f the opt is %f time is %dms\n", diffX, diffE,opt,end-begin);
	alsm_free_all();
}