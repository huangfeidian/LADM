#include "data_generate.h"
#include "../solver/multi_para.h"
#include "../solver/multi_seq.h"
#include "../ALM/PALM/PALM_cuda.h"
#include "../ALM/DALM/DALM_fast.h"
#include <fstream>
#include <iostream>
using namespace std;
using namespace alsm;
template <typename T>
#define EPS1 0.001
#define EPS2 0.05
#define EPS3 0.0001
void test(ofstream& output_file,T* A, T*b,T* output_xG, int m, int n,T target_opt,StopCriteria how_stop)
{

	vector<int> blocks{ 4 };
	auto begin = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<T, std::milli> elapsed;
	T* output = new T[m + n];
	T* output_x = output;
	T* output_e = output+n;
	T* result_b = new T[m];
	
	T* lambda = new T[m];
	stream<DeviceType::CPU> result_stream;
	T norm_b;
	T norm_diff_b;
	T residual_error;
	T current_opt;
	T xdiff_error;
	nrm2<DeviceType::CPU, T>(result_stream, m, b, &norm_b);
#if 1
	int PALM_DALM_stop;
	switch (how_stop)
	{
	case StopCriteria::duality_gap:
		PALM_DALM_stop = 1;
		break;
	case StopCriteria::ground_truth:
		PALM_DALM_stop = -1;
		break;
	case StopCriteria::ground_object:
		PALM_DALM_stop = 6;
		break;
	case StopCriteria::increment:
		PALM_DALM_stop = 5;
		break;
	case StopCriteria::objective_value:
		PALM_DALM_stop = 3;
		break;
	case StopCriteria::kkt_dual_tol:
		PALM_DALM_stop = 7;
		break;
	default:
		cout << "unsupported stop criteria" << endl;
		exit(1);
		break;
	}
#endif 
#if 1
	for (auto block_size : blocks)
	{
#if 0
		memset(output, 0, sizeof(T)*(m + n));
		memset(lambda,0, sizeof(T)*m);
		std::vector<stream<DeviceType::CPU>> cpu_streams=stream<DeviceType::CPU>::create_streams(block_size + 2, false);
		multi_seq<DeviceType::CPU, T> seq_cpu_solver(cpu_streams[block_size + 1], block_size + 1, m, 2000, 10);
		seq_cpu_solver.init_memory();
		seq_cpu_solver.init_server(cpu_streams[block_size + 1], b, lambda, StopCriteria::ground_object, target_opt);
		for (int i = 0; i <block_size; i++)
		{
			seq_cpu_solver.add_client(cpu_streams[i], n / block_size, FunctionObj<T>(UnaryFunc::Abs), A + m*(n / block_size)*i, false, 
				MatrixMemOrd::COL, output_x + (n / block_size)*i, 0, output_xG + (n / block_size)*i);
		}
		seq_cpu_solver.add_client(cpu_streams[block_size], m, FunctionObj<T>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e,0,output_xG+n);
		seq_cpu_solver.init_parameter(0.01, 0.01, m*0.001, 1000, 1.1,0.01);
		begin = std::chrono::high_resolution_clock::now();
		end = std::chrono::high_resolution_clock::now();
		begin = std::chrono::high_resolution_clock::now();
		seq_cpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		output_file << "CPU SEQ" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << seq_cpu_solver.lambda_server.current_eps3 << "," << seq_cpu_solver.lambda_server.total_opt_value << endl;
		alsm_free_all();

		memset(output, 0, sizeof(T)*(m + n));
		memset(lambda, 0, sizeof(T)*m);
		multi_para<DeviceType::CPU, T> para_cpu_solver(cpu_streams[block_size + 1], block_size + 1, m, 2000, 10);
		para_cpu_solver.init_memory();
		para_cpu_solver.init_server(cpu_streams[block_size + 1], b, lambda, StopCriteria::ground_object, target_opt);
		for (int i = 0; i <block_size; i++)
		{
			para_cpu_solver.add_client(cpu_streams[i], n / block_size, FunctionObj<T>(UnaryFunc::Abs), A + m*(n / block_size)*i, false,
				MatrixMemOrd::COL, output_x + (n / block_size)*i, 0, output_xG + (n / block_size)*i);
		}
		para_cpu_solver.add_client(cpu_streams[block_size], m, FunctionObj<T>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e,0,output_xG+n);
		para_cpu_solver.init_parameter(0.01, 0.01, m*0.001, 1000, 1.1, 0.01);
		begin = std::chrono::high_resolution_clock::now();
		para_cpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		output_file << "CPU PARA" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," <<
			seq_cpu_solver.lambda_server.current_eps3 << "," << seq_cpu_solver.lambda_server.total_opt_value << endl;
		alsm_free_all();
		for (auto i : cpu_streams)
		{
			i.destory();
		}
#endif 
#if ALSM_USE_GPU
		//begin gpu
#if 0
		memset(output, 0, sizeof(T)*(m + n));
		memset(lambda,0, sizeof(T)*m);
		std::vector<stream<DeviceType::GPU>> gpu_streams = stream<DeviceType::GPU>::create_streams(block_size + 2, false);
		multi_seq<DeviceType::GPU, T> seq_gpu_solver(gpu_streams[block_size + 1], block_size + 1, m, 2000, 10);
		seq_gpu_solver.init_memory();
		seq_gpu_solver.init_server(gpu_streams[block_size + 1], b, lambda, StopCriteria::ground_object, target_opt);
		for (int i = 0; i <block_size; i++)
		{
			seq_gpu_solver.add_client(gpu_streams[i], n / block_size, FunctionObj<T>(UnaryFunc::Abs), A + m*(n / block_size)*i, false, 
				MatrixMemOrd::COL, output_x + (n / block_size)*i, 0, output_xG + (n / block_size)*i);
		}
		seq_gpu_solver.add_client(gpu_streams[block_size], m, FunctionObj<T>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e,0,output_xG+n);
		seq_gpu_solver.init_parameter(0.01, 0.01,m*0.001, 1000, 1.1,0.01);
		begin = std::chrono::high_resolution_clock::now();
		seq_gpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		output_file << "GPU SEQ" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << seq_gpu_solver.lambda_server.current_eps3 << "," << seq_gpu_solver.lambda_server.total_opt_value << endl;
		alsm_free_all();

		//multi para
		memset(output, 0, sizeof(T)*(m + n));
		memset(lambda,0, sizeof(T)*m);
		multi_para<DeviceType::GPU, T> para_gpu_solver(gpu_streams[block_size + 1], block_size + 1, m, 2000, 10);
		para_gpu_solver.init_memory();
		para_gpu_solver.init_server(gpu_streams[block_size + 1], b, lambda, StopCriteria::ground_object, target_opt);
		for (int i = 0; i <block_size; i++)
		{
			para_gpu_solver.add_client(gpu_streams[i], n / block_size, FunctionObj<T>(UnaryFunc::Abs), A + m*(n / block_size)*i, false,
				MatrixMemOrd::COL, output_x + (n / block_size)*i, 0, output_xG + (n / block_size)*i);
		}
		para_gpu_solver.add_client(gpu_streams[block_size], m, FunctionObj<T>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e,0,output_xG+n);
		para_gpu_solver.init_parameter(0.01, 0.01, m*0.001, 1000, 1.1,0.01);
		begin = std::chrono::high_resolution_clock::now();
		para_gpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		output_file << "GPU PARA" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << para_gpu_solver.lambda_server.current_eps3 << "," << para_gpu_solver.lambda_server.total_opt_value << endl;
		alsm_free_all();
		for (auto i : gpu_streams)
		{
			i.destory();
		}
#endif
#if 1
		//begin multi gpu
		memset(output, 0, sizeof(T)*(m + n));
		memset(lambda,0, sizeof(T)*m);
		memset(result_b, 0, sizeof(T)*m);
		std::vector<stream<DeviceType::GPU>> multi_gpu_streams = stream<DeviceType::GPU>::create_streams(block_size + 2, true);
		multi_seq<DeviceType::GPU, T> multi_seq_gpu_solver(multi_gpu_streams[block_size + 1], block_size + 1, m, 10000, 10);
		multi_seq_gpu_solver.init_memory();
		multi_seq_gpu_solver.init_server(multi_gpu_streams[block_size + 1], b, lambda, how_stop, target_opt);
		for (int i = 0; i <block_size; i++)
		{
			multi_seq_gpu_solver.add_client(multi_gpu_streams[i], n / block_size, FunctionObj<T>(UnaryFunc::Abs), A + m*(n / block_size)*i, false,
				MatrixMemOrd::COL, output_x + (n / block_size)*i,0, output_xG + (n / block_size)*i);
		}
		multi_seq_gpu_solver.add_client(multi_gpu_streams[block_size], m, FunctionObj<T>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e,0,output_xG+n);
		multi_seq_gpu_solver.init_parameter(EPS1, EPS2,1, 10000, 1.1,EPS3);
		begin = std::chrono::high_resolution_clock::now();
		multi_seq_gpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		gemv<DeviceType::CPU, T>(result_stream, MatrixTrans::NORMAL, MatrixMemOrd::COL, m, n, 1.0, A, m, output_x, 0, result_b);
		axpy<DeviceType::CPU, T>(result_stream, m, 1, output_e, result_b);
		axpby<DeviceType::CPU, T>(result_stream, m, 1, b, -1, result_b);
		nrm2<DeviceType::CPU, T>(result_stream, m, result_b, &norm_diff_b);
		residual_error = norm_diff_b / norm_b;
		asum<DeviceType::CPU, T>(result_stream, m + n, output, &current_opt);
		xdiff_error = multi_seq_gpu_solver.lambda_server.epsilon_3;
		output_file << "MULTI GPU SEQ" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << residual_error << 
			","<<xdiff_error<<"," << current_opt << endl;
		cout << "MULTI GPU SEQ" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << residual_error <<
			"," << xdiff_error << "," << current_opt << endl;
		alsm_free_all();
#if 1
		memset(output, 0, sizeof(T)*(m + n));
		memset(lambda,0, sizeof(T)*m);
		multi_para<DeviceType::GPU, T> multi_para_gpu_solver(multi_gpu_streams[block_size + 1], block_size + 1, m, 10000, 10);
		multi_para_gpu_solver.init_memory();
		multi_para_gpu_solver.init_server(multi_gpu_streams[block_size + 1], b, lambda, how_stop, target_opt);
		for (int i = 0; i <block_size; i++)
		{
			multi_para_gpu_solver.add_client(multi_gpu_streams[i], n / block_size, FunctionObj<T>(UnaryFunc::Abs), A + m*(n / block_size)*i, false, 
				MatrixMemOrd::COL, output_x + (n / block_size)*i, 0, output_xG + (n / block_size)*i);
		}
		multi_para_gpu_solver.add_client(multi_gpu_streams[block_size], m, FunctionObj<T>(UnaryFunc::Abs), nullptr, true, MatrixMemOrd::COL, output_e,0,output_xG+n);
		multi_para_gpu_solver.init_parameter(EPS1, EPS2, 1, 10000, 1.1,EPS3);
		begin = std::chrono::high_resolution_clock::now();
		multi_para_gpu_solver.solve();
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - begin;
		gemv<DeviceType::CPU, T>(result_stream, MatrixTrans::NORMAL, MatrixMemOrd::COL, m, n, 1.0, A, m, output_x, 0, result_b);
		axpy<DeviceType::CPU, T>(result_stream, m, 1, output_e, result_b);
		axpby<DeviceType::CPU, T>(result_stream, m, 1, b, -1, result_b);
		nrm2<DeviceType::CPU, T>(result_stream, m, result_b, &norm_diff_b);
		residual_error = norm_diff_b / norm_b;
		asum<DeviceType::CPU, T>(result_stream, m + n, output, &current_opt);
		xdiff_error = multi_para_gpu_solver.lambda_server.epsilon_3;
		output_file << "MULTI GPU para" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << residual_error <<
			"," << xdiff_error << "," << current_opt << endl;
		cout << "MULTI GPU para" << "," << block_size << "," << m << "," << n << "," << elapsed.count() << "," << residual_error <<
			"," << xdiff_error << "," << current_opt << endl;
		alsm_free_all();
#endif
		for (auto i : multi_gpu_streams)
		{
			i.destory();
		}
#endif
#endif

	}
#endif 
	// PLAM
#if 1
	memset(output, 0, sizeof(T)*(m + n));
	memset(lambda, 0, sizeof(T)*m);
	L1Solver_e1x1 PLAM_solver(m, n);
	begin = std::chrono::high_resolution_clock::now();
	PLAM_solver.set_A(A);
	PLAM_solver.solve(b, output_x, output_e, EPS1, 0.00001, 1000, 50, PALM_DALM_stop, output_xG,EPS3);
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - begin;
	T PLAM_eps3 = abs(PLAM_solver.f - PLAM_solver.prev_f) / PLAM_solver.prev_f;
	gemv<DeviceType::CPU, T>(result_stream, MatrixTrans::NORMAL, MatrixMemOrd::COL, m, n, 1.0, A, m, output_x, 0, result_b);
	axpy<DeviceType::CPU, T>(result_stream, m, 1, output_e, result_b);
	axpby<DeviceType::CPU, T>(result_stream, m, 1, b, -1, result_b);
	nrm2<DeviceType::CPU, T>(result_stream, m, result_b, &norm_diff_b);
	residual_error = norm_diff_b / norm_b;
	asum<DeviceType::CPU, T>(result_stream, m + n, output, &current_opt);
	xdiff_error = PLAM_solver.dx / PLAM_solver.nxo;
	
	output_file << "PALM" << ",0 ," << m << "," << n << "," << elapsed.count() << "," << residual_error <<
		"," << xdiff_error << "," << current_opt << endl;
	cout << "PALM" << ",0 ," << m << "," << n << "," << elapsed.count() << "," << residual_error <<
		"," << xdiff_error << "," << current_opt << endl;
	PLAM_solver.free_memory();
#endif
	// DLAM
#if 1
	memset(output, 0, sizeof(T)*(m + n));
	memset(lambda, 0, sizeof(T)*m);
	DALM_solver dalm_solver(m, n+m,PALM_DALM_stop,EPS1,0.000001,EPS3,10000);
	dalm_solver.allocate_memory();
	begin = std::chrono::high_resolution_clock::now();
	
	dalm_solver.solve(output, b, A, output_xG);
	end = std::chrono::high_resolution_clock::now();
	elapsed = end - begin;
	gemv<DeviceType::CPU, T>(result_stream, MatrixTrans::NORMAL, MatrixMemOrd::COL, m, n, 1.0, A, m, output_x, 0, result_b);
	axpy<DeviceType::CPU, T>(result_stream, m, 1, output_e, result_b);
	axpby<DeviceType::CPU, T>(result_stream, m, 1, b, -1, result_b);
	nrm2<DeviceType::CPU, T>(result_stream, m, result_b, &norm_diff_b);
	residual_error = norm_diff_b / norm_b;
	asum<DeviceType::CPU, T>(result_stream, m + n, output, &current_opt);
	xdiff_error = dalm_solver.dx / dalm_solver.nxo;
	output_file << "DALM" << ",0 ,"  << m << "," << n << "," << elapsed.count() << "," << residual_error <<
		"," << xdiff_error<< "," << current_opt << endl;
	cout << "DALM" <<  ",0 ,"  << m << "," << n << "," << elapsed.count() << "," << residual_error <<
		"," << xdiff_error<< "," << current_opt << endl;
	dalm_solver.free_memory();
#endif 
	delete [] output;
	delete [] lambda;
}
int main()
{
	int begin_m = 320;
	int begin_n = 1024 ;
	float* A;
	float* b;
	float* x;
	float* e;
	float* xG;
	float opt_G;
	ofstream output("speedtest.csv");
	output << "type,block_size,m,n,time,residual_eps,xdiff_eps,opt" << endl;
	for (int i = 1; i < 12; i++)
	{
		int m, n;
		m = begin_m*i;
		n = begin_n*i;
		A = new float[m*(n+m)];
		b = new float[m];

		xG = new float[m + n];
		x = xG;
		e = xG+n;
		memset(A, 0, m*n*sizeof(float));
		memset(b, 0, m*sizeof(float));
		memset(e, 0, m*sizeof(float));
		memset(x, 0, n*sizeof(float));
		opt_G=generate_test_data<float>(A, x, e, b, m, n, 0.1);
		output << "opt,0," << m << "," << n << ",0,0,0," << opt_G << endl;
		cout << "opt:"<< opt_G << endl;
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
		test<float>(output, A, b,xG, m, n,opt_G,StopCriteria::kkt_dual_tol);
		delete [] A;
		delete [] b;
		delete [] xG;
		cout << "round " << i << " finished" << endl;

	}
}