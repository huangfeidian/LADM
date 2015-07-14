// this l1 solver is to solve 
//min |x|1 + |e|1 s.t. Ax + e = b\n
#ifndef _H_MULTI_SEQ_H_
#define _H_MULTI_SEQ_H_
#include "alsm_solver.h"
//#include "../lapack/lapack.h"
#include <array>
#include <vector>
namespace alsm
{
	template <DeviceType D, typename T> class multi_seq
	{
	private:
		std::atomic_int ready_thread_count;
		std::vector<std::atomic_bool> update_recieve;
		std::atomic_bool work_finished;
		const std::chrono::microseconds wait_time;
		int x_dimension, b_dimension;
		std::vector<stream<D>> client_streams;
		stream<D> server_stream;
		std::vector<alsm_client<D, T>> all_clients;
		alsm_server<D, T> server;
		int current_client_number;
	private:
		int clients_number;

		//generally the index 0 is for e index 1 is for x and index 2 for server
		std::vector<T*> client_residual;//residual[0] for e residual[1] for x
		T* server_residual;
		std::vector<T*> all_client_x;
		std::vector<T> client_opt;
		T total_opt;//opt[0] for e opt[1] for x opt[2] for server
		T max_eta_norm;
		vector<T> client_eta_norm;
		T* lambda[2];//lambda[0] for server_lambda lambda[1] for lambda_hat
		T server_beta;
		std::vector<T> client_beta;
		std::vector<T*> all_client_A;
		T* b;
		std::vector<T*> output_x;
		std::vector<int> client_dimension;
	public:
		multi_seq(stream<D> in_stream, int in_client_number, int in_b_dimension, int in_max_iter, int in_wait_ms)
			:b_dimension(in_b_dimension), wait_time(in_wait_ms), clients_number(in_client_number), server_stream(in_stream),
			server(ready_thread_count, &update_recieve[0], work_finished, in_clients_number, in_wait_ms, in_max_iter, in_b_dimension, in_stream)
		{
			ready_thread_count.store(0);
			update_recieve = std::vector<std::atomic_bool>(clients_number, false);
			work_finished.store(false);
			client_beta = std::vector<T>(clients_number, 0);
			client_opt = std::vector<T>(clients_number, 0);
			total_opt = 0;
			all_cleint_x = std::vector<T*>(clients_number, nullptr);
			client_residual = std::vector<T*>(clients_number, nullptr);
			eta_norm = std::vector<T>(clients_number, 0);
			all_client_A = std::vector<T*>(clients_number, nullptr);
			client_streams = std::vector<stream<D>>(clients_number, stream<D>());
			current_client_number = 0;
			client_dimension = std::vector<int>(clients_number, 0);
			output_x = std::vector<T*>(clients_number, nullptr);
			// only client can't be null initiated
			all_clients.reserve(clients_number);
		}

	public:
		void add_server(stream<D> in_stream, T* in_b, T* in_lambda)
		{
			server_stream = in_stream;
			alsm_fromcpu<D, T>(server_stream, b, in_b, b_dimension);
			alsm_fromcpu<D, T>(server_stream, lambda[0], in_lambda);
			server.init_problem(client_residual[0], client_opt, &client_eta_norm[0], b, &server_beta, &client_beta[0], lambda[1], lambda[0], server_residual);
		}
		void init_parameter(T in_eps_1, T in_eps_2, T in_beta, T in_max_beta, T in_rho)
		{
			server.init_parameter(in_eps_1, in_eps_2, in_beta, in_max_beta, in_rho);
			server_beta = in_beta;
			for (int i = 0; i < clients_number; i++)
			{
				client_beta[i] = in_beta;
			}
		}
		void add_client(stream<D> in_stream, int in_x_dimension, FunctionObj<T> in_func, T* in_A, bool is_Identity, MatrixMemOrd in_A_ord, T* in_x)
		{
			T* client_x = alsm_malloc<D, T>(3 * in_x_dimension);
			all_client_x[current_client_number] = client_x;
			alsm_fromcpu<D, T>(server_stream, client_x, in_x, in_x_dimension);
			T* client_A;
			T max_sigular = 0;
			if (is_Identity)
			{
				client_A = nullptr;
				max_sigular = b_dimension*clients_number;
			}
			else
			{
				client_A = alsm_malloc<D, T>(in_x_dimension*b_dimension);
				alsm_fromcpu<D, T>(server_stream, client_A, in_A, in_x_dimension*b_dimension);
				nrm2<D, T>(streams[2], in_x_dimension*b_dimension, client_A, &max_sigular);
				max_sigular = max_sigular*max_sigular *clients_number;
			}
			client_dimension[current_client_number] = in_x_dimension;
			output_x[current_client_number] = in_x;
			all_client_A[current_client_number] = client_A;
			client_streams[current_client_number] = in_stream;
			int i = current_client_number;
			alsm_client<D, T> temp_client(work_finished, update_recieve[i], ready_thread_count, wait_time, i, b_dimension, in_x_dimension, in_func);
			temp_client.init_problem(is_Identity, in_A_ord, client_A, client_x, client_x + 2 * in_x_dimension, &client_beta[i], lambda[1], client_residual[i], max_sigular);
			temp_client.connect_server(&client_eta_norm[i], &client_opt[i], client_residual[i], lambda[1]);
			all_clients.push_back(temp_client);
			current_client_number++;
		}
		void init_memory()
		{
			b = alsm_malloc<D, T>(b_dimension);
			T* total_client_residual = alsm_malloc<D, T>(clients_number*b_dimension);
			for (int i = 0; i < clients_number; i++)
			{
				client_residual[i] = total_client_residual + i*b_dimension;
			}
			server_residual = alsm_malloc<D, T>(b_dimension);
			T* total_lambda = alsm_malloc<D, T>(2 * b_dimension);
			lambda[0] = total_lambda;
			lambda[1] = total_lambda + b_dimension;
		}
		void solve()
		{
			T neg_beta = -1 * server_beta;
			axpy<D, T>(server_stream, b_dimension, &neg_beta, b, lambda[1]);//lambda_hat=-beta*b;
			while (!work_finished.load())
			{
				server.send();
				for (auto i : all_clients)
				{
					i.recieve();
					i.compute();
					i.send();
				}
				server.recieve();
				server.compute();
				server.current_iter++;
				if (server.current_iter == server.max_iter)
				{
					fprintf(stdout, " max iteration %d is exceed\n", server.max_iter);
					work_finished.store(true);
				}
			}
			if (server.current_iter % 2)
			{
				for (int i = 0; i < clients_number; i++)
				{
					alsm_tocpu<D, T>(client_streams[i], output_x[i], all_client_x[i] + client_dimension[i], client_dimension[i]);
				}
			}
			else
			{
				for (int i = 0; i < clients_number; i++)
				{
					alsm_tocpu<D, T>(client_streams[i], output_x[i], all_client_x[i], client_dimension[i]);
				}
			}

		}
	};
}
#endif