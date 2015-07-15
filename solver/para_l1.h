// this l1 solver is to solve 
//min |x|1 + |e|1 s.t. Ax + e = b\n
#ifndef __H_PARA_L1_H__
#define __H_PARA_L1_H__
#include "l1_solver.h"

namespace alsm
{
	template <DeviceType D, typename T> class para_l1:public l1_solver<D,T>
	{
	public:
		para_l1(std::array<stream<D>, 3> three_stream, int in_b_dimension, int in_x_dimension, int in_max_iter, int in_wait_ms)
			:l1_solver(three_stream, in_b_dimension, in_x_dimension, in_max_iter, in_wait_ms)
		{

		}
		void solve()
		{
			T neg_beta = -1 * beta[2];
			axpy<D, T>(streams[2], b_dimension, &neg_beta, b, lambda[1]);//lambda_hat=-beta*b;
			ready_thread_count.store(0);
			std::vector<std::thread> all_threads;
			all_threads.emplace_back(&alsm_server<D, T>::work, &server);
			all_threads.emplace_back(&alsm_client<D, T>::work, &e_client);
			all_threads.emplace_back(&alsm_client<D, T>::work, &x_client);
			for (auto& i : all_threads)
			{
				i.join();
			}

			alsm_tocpu<D, T>(streams[0], output_e, e[0], b_dimension);
			alsm_tocpu<D, T>(streams[1], output_x, x[0], x_dimension);

		}
	};
}
#endif