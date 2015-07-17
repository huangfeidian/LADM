#ifndef __H_SEQ_L1_H__
#define __H_SEQ_L1_H__
#include "l1_solver.h"
namespace alsm
{
	template <DeviceType D, typename T> class seq_l1:public l1_solver<D,T>
	{
	public:
		seq_l1(std::array<stream<D>, 3> three_stream, int in_b_dimension, int in_x_dimension, int in_max_iter, int in_wait_ms)
			:l1_solver(three_stream, in_b_dimension, in_x_dimension, in_max_iter, in_wait_ms)
		{

		}
		void solve()
		{
			axpy<D, T>(streams[2], b_dimension, -1*beta[2], b, lambda[1]);//lambda_hat=-beta*b;
			while (!work_finished.load())
			{
				server.send();
				x_client.recieve();
				x_client.compute();
				x_client.send();
				e_client.recieve();
				e_client.compute();
				e_client.send();
				server.recieve();
				server.compute();
				server.current_iter++;
				if (server.current_iter == server.max_iter)
				{
					fprintf(stdout, " max iteration %d is exceed\n", server.max_iter);
					work_finished.store(true);
				}
			}
			alsm_tocpu<D, T>(streams[0], output_e, e[0], b_dimension);
			alsm_tocpu<D, T>(streams[1], output_x, x[0], x_dimension);
		}
	};

};
#endif