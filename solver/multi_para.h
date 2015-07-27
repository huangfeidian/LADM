// this l1 solver is to solve 
//min |x|1 + |e|1 s.t. Ax + e = b\n
#ifndef __H_MULTI_PARA_H__
#define __H_MULTI_PARA_H__
#include "multi_solver.h"
#include <future>
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
			server_stream.set_context();
			init_lambda();
			std::vector<std::future<void>> async_futures(clients_number);
			while (!work_finished.load())
			{
				lambda_server.send();
				for (int i = 0; i < clients_number; i++)
				{
					async_futures[i] = std::async(std::launch::async,&alsm_client<D,T>::task, &all_clients[i]);
				}
				for (auto & i : async_futures)
				{
					i.get();
				}
				lambda_server.recieve();
				lambda_server.compute();
				lambda_server.current_iter++;
				if (lambda_server.current_iter == lambda_server.max_iter)
				{
					fprintf(stdout, " max iteration %d is exceed\n", lambda_server.max_iter);
					work_finished.store(true);
				}
			}

			for (int i = 0; i < clients_number; i++)
			{
				client_streams[i].set_context();
				tocpu<D, T>(client_streams[i], output_x[i], device_x[i], clients_dimension[i]);
			}
			server_stream.set_context();
			tocpu<D, T>(server_stream, output_lambda, lambda[0], b_dimension);

			
		}
	};
}
#endif