#ifndef __H_MULTI_SOLVER_H__
#define __H_MULTI_SOLVER_H__
#include "alsm_client.h"
#include "alsm_server.h"
//#include "../lapack/lapack.h"
#include <array>
#include <vector>
namespace alsm
{
	namespace
	{
		
	}
	template <DeviceType D, typename T> class multi_solver
	{
	public:
		std::atomic_int ready_thread_count;
		std::atomic_bool work_finished;
		const std::chrono::microseconds wait_time;
		int  b_dimension;
		std::vector<stream<D>> client_streams;
		std::array<cache_align_storage<std::atomic_bool>,20> all_client_turns;
		//this program only support less than 20 blocks, you can change 20 to any number you like to support more
		stream<D> server_stream;
		std::vector<alsm_client<D, T>> all_clients;
		alsm_server<D, T> lambda_server;
		int current_client_number;
		int clients_number;
		std::vector<int> clients_dimension;
	public:
		//these four variables for vectors are synced between devices
		//generally the index 0 is for e index 1 is for x and index 2 for server
		std::vector<T*> clients_residual;//residual[0] for e residual[1] for x
		std::vector<T*> device_residual;//residual[0] for e residual[1] for x
		std::vector<T*> device_lambda;
		T* lambda[2];//lambda[0] for server_lambda lambda[1] for lambda_hat
		

	public:
		T* total_residual;
		StopCriteria stop_type;
		std::vector<T> clients_opt;
		std::vector<T> clients_x_diff_nrm2;
		std::vector<T> clients_x_old_nrm2;
		std::vector<T> clients_xG_diff_nrm2;
		T total_residual_nrm2;
		T server_beta;
		std::vector<T> clients_beta;
	public:
		//these memories are used for input and output
		std::vector<T*> device_x;
		std::vector<T*> clients_A;
		T* b;
		std::vector<T*> output_x;
		T* output_lambda;
		
	public:
		multi_solver(stream<D> in_stream, int in_client_number, int in_b_dimension, int in_max_iter, int in_wait_ms)
			:b_dimension(in_b_dimension), wait_time(in_wait_ms), clients_number(in_client_number), server_stream(in_stream), 
			lambda_server(&ready_thread_count, &all_client_turns[0],&work_finished, in_client_number, in_wait_ms, in_max_iter, in_b_dimension, in_stream)
		{
			ready_thread_count.store(0);
			for (int i = 0; i < clients_number; i++)
			{
				all_client_turns[i].data.store(false);
			}
			work_finished.store(false);
			clients_beta = std::vector<T>(clients_number, 0);
			clients_opt = std::vector<T>(clients_number, 0);
			device_x = std::vector<T*>(clients_number, nullptr);
			clients_residual = std::vector<T*>(clients_number, nullptr);
			device_residual = std::vector<T*>(clients_number, nullptr);
			clients_x_diff_nrm2 = std::vector<T>(clients_number, 0);
			clients_x_old_nrm2 = std::vector<T>(clients_number, 0);
			clients_xG_diff_nrm2 = std::vector<T>(clients_number, 0);
			clients_A = std::vector<T*>(clients_number, nullptr);
			client_streams = std::vector<stream<D>>(clients_number, stream<D>());
			current_client_number = 0;
			clients_dimension = std::vector<int>(clients_number, 0);
			output_x = std::vector<T*>(clients_number, nullptr);
			device_lambda = std::vector<T*>(clients_number, nullptr);
			// only client can't be null initiated
			all_clients.reserve(clients_number);
			
		}

	public:
		void init_memory()
		{
			server_stream.set_context();
			b = alsm_malloc<D, T>(server_stream, b_dimension);
			alsm_memset<D, T>(b, 0, b_dimension);
			T* total_client_residual = alsm_malloc<D, T>(server_stream,clients_number*b_dimension);
			alsm_memset<D, T>(total_client_residual, 0, clients_number*b_dimension);
			for (int i = 0; i < clients_number; i++)
			{
				clients_residual[i] = total_client_residual + i*b_dimension;
			}
			total_residual = alsm_malloc<D, T>(server_stream, b_dimension);
			alsm_memset<D, T>(total_residual, 0, b_dimension);
			T* total_lambda = alsm_malloc<D, T>(server_stream, 2 * b_dimension);
			alsm_memset<D, T>(total_lambda, 0, 2 * b_dimension);
			lambda[0] = total_lambda;
			lambda[1] = total_lambda + b_dimension;
		}
		void init_server(stream<D> in_stream, T* in_b, T* in_lambda,StopCriteria in_stop_type,T target_opt=-1)
		{
			output_lambda = in_lambda;
			server_stream = in_stream;
			stop_type = in_stop_type;
			fromcpu<D, T>(server_stream, b, in_b, b_dimension);
			fromcpu<D, T>(server_stream, lambda[0], in_lambda,b_dimension);
			lambda_server.init_problem(b, lambda[1], lambda[0], total_residual,in_stop_type,target_opt);
		}
		void init_parameter(T in_eps_1, T in_eps_2, T in_beta, T in_max_beta, T in_rho,T in_eps_3)
		{
			lambda_server.init_parameter(in_eps_1, in_eps_2, in_beta, in_max_beta, in_rho,in_eps_3);
			server_beta = in_beta;
			for (int i = 0; i < clients_number; i++)
			{
				clients_beta[i] = in_beta;
			}
		}
		void add_client(stream<D> in_stream, int in_x_dimension, FunctionObj<T> in_func, T* in_A, bool is_Identity, MatrixMemOrd in_A_ord, T* in_x,T in_eta=0,T* in_xG=nullptr)
		{
			in_stream.set_context();
			T* client_x = alsm_malloc<D, T>(in_stream,3 * in_x_dimension);
			T* client_xG = nullptr;
			if (in_xG == nullptr)
			{
				client_xG = nullptr;
			}
			else
			{
				client_xG = alsm_malloc<D, T>(in_stream, in_x_dimension);
				fromcpu<D, T>(in_stream, client_xG, in_xG, in_x_dimension);
			}
			
			T* current_device_residual = alsm_malloc<D, T>(in_stream,b_dimension);
			T* current_device_lambda = alsm_malloc<D, T>(in_stream, b_dimension);
			device_residual[current_client_number] = current_device_residual;
			device_lambda[current_client_number] = current_device_lambda;
			alsm_memset<D, T>(client_x, 0, 3 * in_x_dimension);
			fromcpu<D, T>(in_stream, client_x, in_x, in_x_dimension);
			T* client_A;
			T inited_eta = 0;
			if (is_Identity)
			{
				client_A = nullptr;
				if (in_eta == 0)
				{
					inited_eta =  1;
				}

				
				copy<D, T>(in_stream, b_dimension, client_x, current_device_residual);
				to_server<D, T>(in_stream, clients_residual[current_client_number], current_device_residual,b_dimension, server_stream.device_index);
				in_stream.sync();
			}
			else
			{
				client_A = alsm_malloc<D, T>(in_stream, in_x_dimension*b_dimension);
				fromcpu<D, T>(in_stream, client_A, in_A, in_x_dimension*b_dimension);
				int lda = (in_A_ord == MatrixMemOrd::ROW) ?in_x_dimension : b_dimension;
				gemv<D, T>(in_stream, MatrixTrans::NORMAL, in_A_ord, b_dimension, in_x_dimension, 1, client_A, lda, 
					client_x, 0, current_device_residual);//residual=A*x_1
				to_server<D, T>(in_stream, clients_residual[current_client_number], current_device_residual,b_dimension, server_stream.device_index);
				if (in_eta == 0)
				{
					nrm2<D, T>(in_stream, b_dimension*in_x_dimension, client_A, &inited_eta);
				}

			}

			in_stream.sync();
			if (in_eta == 0)
			{
				inited_eta *= inited_eta;
				inited_eta *= clients_number ;
			}
			else
			{
				inited_eta = in_eta;
			}
			device_x[current_client_number] = client_x;
			clients_dimension[current_client_number] = in_x_dimension;
			output_x[current_client_number] = in_x;
			clients_A[current_client_number] = client_A;
			client_streams[current_client_number] = in_stream;
			int i = current_client_number;
			alsm_client<D, T> temp_client(&work_finished, &all_client_turns[i], &ready_thread_count, wait_time.count(), i, b_dimension, in_x_dimension, in_func, in_stream);
			temp_client.init_problem(is_Identity, in_A_ord, client_A, client_x, client_x + 2 * in_x_dimension, &clients_beta[i], device_lambda[i], device_residual[i], inited_eta,stop_type,client_xG );
			temp_client.connect_server(server_stream.device_index, &clients_opt[i], clients_residual[i], &clients_x_old_nrm2[i], &clients_x_diff_nrm2[i], &clients_xG_diff_nrm2[i]);
			lambda_server.add_client(&clients_opt[i], &clients_beta[i],inited_eta, clients_residual[i], device_lambda[i],&clients_x_old_nrm2[i],&clients_x_diff_nrm2[i], in_stream,  &clients_xG_diff_nrm2[i]);
			all_clients.push_back(temp_client);
			current_client_number++;
		}
		void init_lambda()
		{
			server_stream.set_context();
			for (auto i : clients_residual)
			{
				axpy<D, T>(server_stream, b_dimension, 1, i, total_residual);
			}
			axpy<D, T>(server_stream, b_dimension, -1, b, total_residual);
			axpy<D, T>(server_stream, b_dimension, server_beta, total_residual, lambda[1]);//lambda_hat=-beta*b;
			server_stream.sync();
		}
		virtual void solve() = 0;
	};

}
#endif