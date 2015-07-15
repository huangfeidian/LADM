// this l1 solver is to solve 
//min |x|1 + |e|1 s.t. Ax + e = b\n
#ifndef __H_MULTI_SEQ_H__
#define __H_MULTI_SEQ_H__
#include "multi_solver.h"

namespace alsm
{
	template <DeviceType D, typename T> class multi_seq:public multi_solver<D,T>
	{
	public:
		multi_seq(stream<D> in_stream, int in_client_number, int in_b_dimension, int in_max_iter, int in_wait_ms)
			:multi_solver(in_stream, in_client_number, in_b_dimension, in_max_iter, in_wait_ms)
		{

		}
		void solve()
		{
			init_lambda();
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

			for (int i = 0; i < clients_number; i++)
			{
				alsm_tocpu<D, T>(client_streams[i], output_x[i], clients_x[i], clients_dimension[i]);
			}
		}
	};
}
#endif