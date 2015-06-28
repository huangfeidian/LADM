#include <random>
#include <functional>
#include <vector>
#include "solver/alsm_solver.h"
#include <random>
#include <vector>
using namespace alsm;
using namespace std;
#define ARRAY_SIZE 100
#define CLIENT_NUM 4
int main()
{
	double* host_b;
	double* host_x;
	double* host_lambda_hat;
	double* host_residual;
	double* total_residual;
	double* host_lambda;
	double* client_x[CLIENT_NUM];//for x_1 x_2
	double client_rho[CLIENT_NUM];
	double client_opt[CLIENT_NUM];
	double client_beta[CLIENT_NUM];
	double* client_lambda[CLIENT_NUM];
	double* client_v[CLIENT_NUM];
	double client_eta_nrm[CLIENT_NUM];
	double* client_residual[CLIENT_NUM];
	double* Matrix[CLIENT_NUM];
	double beta=0.01;
	//FILE* debug_file = fopen("train.csv", "wb");
	mt19937 seed;
	normal_distribution<double> dst(10, 10);
	host_x = alsm_malloc<DeviceType::CPU,double>(ARRAY_SIZE*CLIENT_NUM);
	alsm_memset<DeviceType::CPU, double>(host_x, 0, ARRAY_SIZE*CLIENT_NUM);
	host_lambda = alsm_malloc<DeviceType::CPU, double>(ARRAY_SIZE);
	alsm_memset<DeviceType::CPU, double>(host_lambda, 0, ARRAY_SIZE);
	host_lambda_hat = alsm_malloc<DeviceType::CPU, double>(ARRAY_SIZE);
	alsm_memset<DeviceType::CPU, double>(host_lambda_hat, 0, ARRAY_SIZE);
	host_residual = alsm_malloc<DeviceType::CPU, double>(ARRAY_SIZE*CLIENT_NUM);
	alsm_memset<DeviceType::CPU, double>(host_residual, 0, ARRAY_SIZE*CLIENT_NUM);
	total_residual = alsm_malloc<DeviceType::CPU, double>(ARRAY_SIZE);
	alsm_memset<DeviceType::CPU, double>(total_residual, 0, ARRAY_SIZE);
	host_b = alsm_malloc<DeviceType::CPU, double>(ARRAY_SIZE);
	alsm_memset<DeviceType::CPU, double>(host_b, 0, ARRAY_SIZE);
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		host_b[i] = dst(seed);
		total_residual[i] = -host_b[i];
		host_lambda_hat[i] = -1*beta*host_b[i];
	}
	client_lambda[0] = alsm_malloc<DeviceType::CPU, double>(CLIENT_NUM* ARRAY_SIZE);
	alsm_memset<DeviceType::CPU, double>(client_lambda[0], 0, ARRAY_SIZE*CLIENT_NUM);
	client_lambda[1] = client_lambda[0] + ARRAY_SIZE;
	client_lambda[2] = client_lambda[1] + ARRAY_SIZE;
	client_lambda[3] = client_lambda[2] + ARRAY_SIZE;
	client_v[0] = alsm_malloc<DeviceType::CPU, double>(CLIENT_NUM * ARRAY_SIZE);
	alsm_memset<DeviceType::CPU, double>(client_v[0], 0, ARRAY_SIZE*CLIENT_NUM);
	client_v[1] = client_v[0] + ARRAY_SIZE;
	client_v[2] = client_v[1] + ARRAY_SIZE;
	client_v[3] = client_v[2] + ARRAY_SIZE;
	client_residual[0] = alsm_malloc<DeviceType::CPU, double>(CLIENT_NUM* ARRAY_SIZE);
	alsm_memset<DeviceType::CPU, double>(client_residual[0], 0, ARRAY_SIZE*CLIENT_NUM);
	client_residual[1] = client_residual[0] + ARRAY_SIZE;
	client_residual[2] = client_residual[1] + ARRAY_SIZE;
	client_residual[3] = client_residual[2] + ARRAY_SIZE;
	client_x[0] = alsm_malloc<DeviceType::CPU, double>(ARRAY_SIZE*8);//multiplied by 4 because we pack x_1 x_2 togather
	alsm_memset<DeviceType::CPU, double>(client_x[0], 0, ARRAY_SIZE * 8);
	client_x[1] = client_x[0]+2*ARRAY_SIZE;
	client_x[2] = client_x[1] + 2 * ARRAY_SIZE;
	client_x[3] = client_x[2] + 2*ARRAY_SIZE;
	Matrix[0] = alsm_malloc<DeviceType::CPU, double>(ARRAY_SIZE*ARRAY_SIZE*4);
	alsm_memset<DeviceType::CPU, double>(Matrix[0], 0, 4 * ARRAY_SIZE*ARRAY_SIZE);
	Matrix[1] = Matrix[0] + ARRAY_SIZE*ARRAY_SIZE;
	Matrix[2] = Matrix[1] + ARRAY_SIZE*ARRAY_SIZE;
	Matrix[3] = Matrix[2] + ARRAY_SIZE*ARRAY_SIZE;
	stream<DeviceType::CPU> cpu_stream_1;
	stream<DeviceType::CPU> cpu_stream_2;
	stream<DeviceType::CPU> cpu_stream_3;
	stream<DeviceType::CPU> cpu_stream_4;
	stream<DeviceType::CPU> cpu_stream_5;
	std::atomic_bool work_finished;
	work_finished.store(false);
	std::atomic_bool update_recieved[4];
	for (int i = 0; i < 4; i++)
	{
		update_recieved[i].store(false);
	}
	std::atomic_int free_thread_count;
	free_thread_count.store(0);
	alsm_server<DeviceType::CPU,double> server(free_thread_count, update_recieved, work_finished,CLIENT_NUM,2,100,ARRAY_SIZE ,cpu_stream_5);
	alsm_client<DeviceType::CPU, double> cpu_client_1(work_finished, update_recieved[0], free_thread_count,1,ARRAY_SIZE,ARRAY_SIZE,FunctionObj<double>(UnaryFunc::Abs),cpu_stream_1);
	alsm_client<DeviceType::CPU, double> cpu_client_2(work_finished, update_recieved[1], free_thread_count,2, ARRAY_SIZE, ARRAY_SIZE, FunctionObj<double>(UnaryFunc::Abs), cpu_stream_2);
	alsm_client<DeviceType::CPU, double> cpu_client_3(work_finished, update_recieved[2], free_thread_count, 3,ARRAY_SIZE, ARRAY_SIZE, FunctionObj<double>(UnaryFunc::Abs), cpu_stream_3);
	alsm_client<DeviceType::CPU, double> cpu_client_4(work_finished, update_recieved[3], free_thread_count,4, ARRAY_SIZE, ARRAY_SIZE, FunctionObj<double>(UnaryFunc::Abs), cpu_stream_4);

	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		host_b[i] = dst(seed);
	}
	for (int j = 0; j < 4; j++)
	{
		for (int i = 0; i < ARRAY_SIZE; i++)
		{
			Matrix[j][i*ARRAY_SIZE + i] = 1;
		}
	}
	server.init_problem(host_residual, client_opt, client_eta_nrm, host_b, &beta,client_beta, host_lambda_hat, host_lambda, total_residual);
	server.init_parameter(0.001, 0.001,0.01, 100, 1.2);
	//server.set_debug_file(debug_file);
	cpu_client_1.init_problem(true,MatrixMemOrd::COL,Matrix[0],  client_x[0], client_v[0], client_beta + 0,client_lambda[0],client_residual[0],5);
	cpu_client_2.init_problem(true, MatrixMemOrd::COL, Matrix[1], client_x[1], client_v[1], client_beta + 1, client_lambda[1], client_residual[1], 5);
	cpu_client_3.init_problem(true, MatrixMemOrd::COL, Matrix[2], client_x[2], client_v[2], client_beta + 2, client_lambda[2], client_residual[2], 5);
	cpu_client_4.init_problem(true, MatrixMemOrd::COL, Matrix[3], client_x[3], client_v[3], client_beta + 3, client_lambda[3], client_residual[3], 5);
	cpu_client_1.connect_server(client_eta_nrm + 0, client_opt + 0, host_residual + 0*ARRAY_SIZE, host_lambda_hat );
	cpu_client_2.connect_server(client_eta_nrm + 1, client_opt + 1, host_residual + 1 * ARRAY_SIZE, host_lambda_hat );
	cpu_client_3.connect_server(client_eta_nrm + 2, client_opt + 2, host_residual + 2 * ARRAY_SIZE, host_lambda_hat );
	cpu_client_4.connect_server(client_eta_nrm + 3, client_opt + 3, host_residual + 3 * ARRAY_SIZE, host_lambda_hat );
	thread host_thread([&]()
	{
		server.work();
	});
	vector<thread> client_thread_vector;
	client_thread_vector.reserve(4);
	client_thread_vector.emplace_back(&alsm_client<DeviceType::CPU, double>::work, &cpu_client_1);
	client_thread_vector.emplace_back(&alsm_client<DeviceType::CPU, double>::work, &cpu_client_2);
	client_thread_vector.emplace_back(&alsm_client<DeviceType::CPU, double>::work, &cpu_client_3);
	client_thread_vector.emplace_back(&alsm_client<DeviceType::CPU, double>::work, &cpu_client_4);
	host_thread.join();
	for (auto& i : client_thread_vector)
	{
		i.join();
	}
	//fclose(debug_file);
	alsm_free_all();


}