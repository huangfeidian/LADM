// this l1 solver is to solve 
//min |x|1 + |e|1 s.t. Ax + e = b\n
#ifndef __H_MULTI_PARA_H__
#define __H_MULTI_PARA_H__
#include "multi_solver.h"
namespace alsm
{
	template <DeviceType D, typename T> class multi_para:public multi_solver<D,T>
	{
	
	public:
		multi_para(stream<D> in_stream, int in_client_number, int in_b_dimension, int in_max_iter, int in_wait_ms)
			:multi_solver(in_stream, in_client_number, in_b_dimension, in_max_iter, in_wait_ms)
		{

		}
		void solve()
		{
			T neg_beta = -1 * server_beta;
			axpy<D, T>(server_stream, b_dimension, &neg_beta, b, lambda[1]);//lambda_hat=-beta*b;
			std::vector<std::thread> all_threads;
			all_threads.emplace_back(&alsm_server<D, T>::work, &server);
			for (auto i : all_clients)
			{
				all_threads.emplace_back(&alsm_client<D, T>::work, &i);
			}
			for (auto& i : all_threads)
			{
				i.join();
			}
			if (server.current_iter % 2)
			{
				for (int i = 0; i < clients_number; i++)
				{
					alsm_tocpu<D, T>(client_streams[i], output_x[i], clients_x[i]+clients_dimension[i], clients_dimension[i]);
				}
			}
			else
			{
				for (int i = 0; i < clients_number; i++)
				{
					alsm_tocpu<D, T>(client_streams[i], output_x[i], clients_x[i], clients_dimension[i]);
				}
			}
			
		}
	};
}
#endif