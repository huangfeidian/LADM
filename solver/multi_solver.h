#ifndef __H_MULTI_SOLVER_H__
#define __H_MULTI_SOLVER_H__
#include "alsm_client.h"
#include "alsm_server.h"
//#include "../lapack/lapack.h"
#include <array>
#include <vector>
namespace alsm
{
	template <DeviceType D, typename T> class multi_solver
	{
	public:
		std::atomic_int ready_thread_count;
		std::atomic_bool work_finished;
		const std::chrono::microseconds wait_time;
		int  b_dimension;
		std::vector<stream<D>> client_streams;
		std::array<cache_align_storage<std::atomic_bool>,20> all_client_turns;
		stream<D> server_stream;
		std::vector<alsm_client<D, T>> all_clients;
		alsm_server<D, T> server;
		int current_client_number;
	public:
		int clients_number;

		//generally the index 0 is for e index 1 is for x and index 2 for server
		std::vector<T*> clients_residual;//residual[0] for e residual[1] for x
		T* server_residual;
		std::vector<T*> clients_x;
		std::vector<T> clients_opt;
		T total_opt;//opt[0] for e opt[1] for x opt[2] for server
		T max_eta_norm;
		std::vector<T> clients_eta_norm;
		T* lambda[2];//lambda[0] for server_lambda lambda[1] for lambda_hat
		T server_beta;
		std::vector<T> clients_beta;
		std::vector<T*> clients_A;
		T* b;
		std::vector<T*> output_x;
		std::vector<int> clients_dimension;
	public:
		multi_solver(stream<D> in_stream, int in_client_number, int in_b_dimension, int in_max_iter, int in_wait_ms)
			:b_dimension(in_b_dimension), wait_time(in_wait_ms), clients_number(in_client_number), server_stream(in_stream), 
			server(&ready_thread_count, &all_client_turns[0],&work_finished, in_client_number, in_wait_ms, in_max_iter, in_b_dimension, in_stream)
		{
			ready_thread_count.store(0);
			for (int i = 0; i < clients_number; i++)
			{
				all_client_turns[i].data.store(false);
			}
			work_finished.store(false);
			clients_beta = std::vector<T>(clients_number, 0);
			clients_opt = std::vector<T>(clients_number, 0);
			total_opt = 0;
			clients_x = std::vector<T*>(clients_number, nullptr);
			clients_residual = std::vector<T*>(clients_number, nullptr);
			clients_eta_norm = std::vector<T>(clients_number, 0);
			clients_A = std::vector<T*>(clients_number, nullptr);
			client_streams = std::vector<stream<D>>(clients_number, stream<D>());
			current_client_number = 0;
			clients_dimension = std::vector<int>(clients_number, 0);
			output_x = std::vector<T*>(clients_number, nullptr);
			// only client can't be null initiated
			all_clients.reserve(clients_number);
		}

	public:
		void init_memory()
		{
			b = alsm_malloc<D, T>(b_dimension);
			alsm_memset<D, T>(b, 0, b_dimension);
			T* total_client_residual = alsm_malloc<D, T>(clients_number*b_dimension);
			alsm_memset<D, T>(total_client_residual, 0, clients_number*b_dimension);
			for (int i = 0; i < clients_number; i++)
			{
				clients_residual[i] = total_client_residual + i*b_dimension;
			}
			server_residual = alsm_malloc<D, T>(b_dimension);
			alsm_memset<D, T>(server_residual, 0, b_dimension);
			T* total_lambda = alsm_malloc<D, T>(2 * b_dimension);
			alsm_memset<D, T>(total_lambda, 0, 2 * b_dimension);
			lambda[0] = total_lambda;
			lambda[1] = total_lambda + b_dimension;
		}
		void init_server(stream<D> in_stream, T* in_b, T* in_lambda)
		{
			server_stream = in_stream;
			alsm_fromcpu<D, T>(server_stream, b, in_b, b_dimension);
			alsm_fromcpu<D, T>(server_stream, lambda[0], in_lambda,b_dimension);
			server.init_problem(clients_residual[0], &clients_opt[0], &clients_eta_norm[0], b, &server_beta, &clients_beta[0], lambda[1], lambda[0], server_residual);
		}

		void init_parameter(T in_eps_1, T in_eps_2, T in_beta, T in_max_beta, T in_rho)
		{
			server.init_parameter(in_eps_1, in_eps_2, in_beta, in_max_beta, in_rho);
			server_beta = in_beta;
			for (int i = 0; i < clients_number; i++)
			{
				clients_beta[i] = in_beta;
			}
		}
		void add_client(stream<D> in_stream, int in_x_dimension, FunctionObj<T> in_func, T* in_A, bool is_Identity, MatrixMemOrd in_A_ord, T* in_x)
		{
			T* client_x = alsm_malloc<D, T>(3 * in_x_dimension);
			alsm_memset<D, T>(client_x, 0, 3 * in_x_dimension);
			alsm_fromcpu<D, T>(server_stream, client_x, in_x, in_x_dimension);
			T* client_A;
			T max_sigular = 0;
			if (is_Identity)
			{
				client_A = nullptr;
				max_sigular = clients_number;
				copy<D, T>(server_stream, b_dimension, client_x, clients_residual[current_client_number]);
			}
			else
			{
				client_A = alsm_malloc<D, T>(in_x_dimension*b_dimension);

				alsm_fromcpu<D, T>(server_stream, client_A, in_A, in_x_dimension*b_dimension);
				nrm2<D, T>(server_stream, in_x_dimension*b_dimension, client_A, &max_sigular);
				max_sigular = max_sigular*max_sigular *clients_number;
				int lda = (in_A_ord == MatrixMemOrd::ROW) ?in_x_dimension : b_dimension;
				gemv<D, T>(server_stream, MatrixTrans::NORMAL, in_A_ord, b_dimension, in_x_dimension, 1, client_A, lda, 
					client_x, 0, clients_residual[current_client_number]);//residual=A*x_1
			}
			clients_x[current_client_number] = client_x;
			clients_dimension[current_client_number] = in_x_dimension;
			output_x[current_client_number] = in_x;
			clients_A[current_client_number] = client_A;
			client_streams[current_client_number] = in_stream;
			int i = current_client_number;
			alsm_client<D, T> temp_client(&work_finished, &all_client_turns[i], &ready_thread_count, wait_time.count(), i, b_dimension, in_x_dimension, in_func, in_stream);
			temp_client.init_problem(is_Identity, in_A_ord, client_A, client_x, client_x + 2 * in_x_dimension, &clients_beta[i], lambda[1], clients_residual[i], max_sigular);
			temp_client.connect_server(&clients_eta_norm[i], &clients_opt[i], clients_residual[i], lambda[1]);
			all_clients.push_back(temp_client);
			current_client_number++;
		}
		void init_lambda()
		{
			for (auto i : clients_residual)
			{
				axpy<D, T>(server_stream, b_dimension, 1, i, server_residual);
			}
			axpy<D, T>(server_stream, b_dimension, -1, b, server_residual);
			axpy<D, T>(server_stream, b_dimension, server_beta, server_residual, lambda[1]);//lambda_hat=-beta*b;
			
		}
		virtual void solve() = 0;
	};

}
#endif