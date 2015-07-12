// this l1 solver is to solve 
//min |x|1 + |e|1 s.t. Ax + e = b\n
#ifndef _H_L1_MIN_SOLVER_H_
#define _H_L1_MIN_SOLVER_H_
#include "alsm_solver.h"
//#include "../lapack/lapack.h"
#include <array>
namespace alsm
{
	template <DeviceType D, typename T> class l1_solver
	{
	private:
		std::atomic_int ready_thread_count;
		std::array<std::atomic_bool, 2> update_recieve;
		std::atomic_bool work_finished;
		const std::chrono::microseconds wait_time;
		int x_dimension, b_dimension;
		std::array<stream<D>, 3> streams;
		alsm_client<D, T> e_client;
		alsm_client<D, T> x_client;
		alsm_server<D, T> server;
	private:
		//generally the index 0 is for e index 1 is for x and index 2 for server
		T* client_residual[2];//residual[0] for e residual[1] for x
		T* server_residual;

		T* x[3];//x[2]=v_x
		T* e[3];//e[2]=v_e
		T opt[3];//opt[0] for e opt[1] for x opt[2] for server
		T eta_norm[3];//eta_norm[0] for e eta_norm[1] for x eta_norm[2] for max_eta_norm
		T* lambda[2];//lambda[0] for server_lambda lambda[1] for lambda_hat
		T beta[3];//beta[0] for e beta[1] for x beta[2] for the server beta
		// the memory below has been allocated before we create the solver
		T* A;
		T* b;
		T* output_x;
		T* output_e;
	public:
		l1_solver(std::array<stream<D>, 3> three_stream, int in_b_dimension, int in_x_dimension, int in_max_iter, int in_wait_ms)
			: x_dimension(in_x_dimension), b_dimension(in_b_dimension), wait_time(in_wait_ms), streams(three_stream),
			e_client(work_finished, update_recieve[0], ready_thread_count, in_wait_ms,0, in_b_dimension, in_b_dimension, FunctionObj<T>(UnaryFunc::Abs), three_stream[0]),
			x_client(work_finished, update_recieve[1], ready_thread_count,in_wait_ms, 1, in_b_dimension, in_x_dimension, FunctionObj<T>(UnaryFunc::Abs), three_stream[1]),
			server(ready_thread_count, &update_recieve[0], work_finished,2, in_wait_ms, in_max_iter, in_b_dimension, three_stream[2])
		{
			ready_thread_count.store(0);
			update_recieve[0].store(false);
			update_recieve[1].store(false);
			work_finished.store(false);
			beta[0] = beta[1]= beta[2] = 0;
			eta_norm[0] = eta_norm[1] = eta_norm[2] = 0;
		}

	public:
		void init_memory()
		{
			A = alsm_malloc<D, T>(b_dimension*x_dimension);
			b = alsm_malloc<D, T>(b_dimension);
			client_residual[0] = alsm_malloc<D, T>(b_dimension * 2);
			alsm_memset<D, T>(client_residual[0], 0, 2 * b_dimension);
			client_residual[1] = client_residual[0] + b_dimension;
			server_residual = alsm_malloc<D, T>(b_dimension);
			alsm_memset<D, T>(server_residual, 0, b_dimension);
			x[0] = alsm_malloc<D, T>(3 * x_dimension);
			alsm_memset<D, T>(x[0], 0, 3 * x_dimension);
			x[1] = x[0] + x_dimension;
			x[2] = x[1] + x_dimension;
			e[0] = alsm_malloc<D, T>(3 * b_dimension);
			alsm_memset<D, T>(e[0], 0, 3 * b_dimension);
			e[1] = e[0] + b_dimension;
			e[2] = e[1] + b_dimension;
			lambda[0] = alsm_malloc<D, T>(b_dimension * 2);
			alsm_memset<D, T>(lambda[0], 0, 2 * b_dimension);
			lambda[1] = lambda[0] + b_dimension;
		}
		void init_parameter(T in_eps_1, T in_eps_2,T in_beta, T in_max_beta, T in_rho )
		{
			server.init_parameter(in_eps_1, in_eps_2,in_beta, in_max_beta, in_rho);
			beta[0] = beta[1] = beta[2] = in_beta;
		}
		void init_problem(MatrixMemOrd in_A_ord,T* in_A, T* in_b, T* in_output_x, T* in_output_e)
		{
			T alpha_one = 1.0;
			T alpha_neg_one = -1.0;
			T alpha_half = 0.5;
			T max_sigular = 0.0;
			alsm_fromcpu<D, T>(streams[2], A, in_A, b_dimension*x_dimension);
			alsm_fromcpu<D, T>(streams[2], b, in_b, b_dimension);
			int lda = (in_A_ord == MatrixMemOrd::ROW) ? x_dimension : b_dimension;
			output_x = in_output_x;
			output_e = in_output_e;
			alsm_memset<D, T>(lambda[0], 0, 2*b_dimension);//lambda=0,lambda_hat=0;
			alsm_memset<D, T>(server_residual, 0, b_dimension);
			// a initial guess e by set e=1/2*b;
			//axpy<D, T>(streams[2],b_dimension, &alpha_half, in_b, e[0]);
			alsm_memset<D, T>(e[0], 0, b_dimension);//e=0
			//alsm_memcpy<D, T>(streams[2], server_residual, e[0], b_dimension);
			// a initial guess x by solving min(|2Ax-b|_2)
			//gels<D, T>(streams[2], in_A_ord, MatrixTrans::NORMAL, b_dimension, x_dimension, in_A, lda, e[0], x[0]);
			alsm_memset<D, T>(x[0], 0, x_dimension);//x=0
			//residual=Ax[0]+e[0]-b
			//gemv<D, T>(streams[2], MatrixTrans::NORMAL, in_A_ord, b_dimension, x_dimension, &alpha_one, in_A, lda, x[0], &alpha_one, server_residual);
			
			
			//max_sigular=norm2(A)^2*2
			//svds_max<D,T>(streams[2], in_A_ord, b_dimension, x_dimension, A, lda, &max_sigular);
			nrm2<D, T>(streams[2], x_dimension*b_dimension, A, &max_sigular);
			max_sigular = max_sigular*max_sigular * 2;
			//max_sigular = 20;
			e_client.init_problem(true,in_A_ord,A, e[0], e[2], &beta[0], lambda[1], client_residual[0],2);//for e the sigular is 1 so we just make it 3
			x_client.init_problem(false,in_A_ord,A, x[0], x[2], &beta[1], lambda[1], client_residual[1],max_sigular);
			server.init_problem(client_residual[0], opt, eta_norm, b, &beta[2] , &beta[0], lambda[1], lambda[0], server_residual);
			e_client.connect_server(eta_norm + 0, opt + 0, client_residual[0], lambda[1]);
			x_client.connect_server(eta_norm + 1, opt + 1, client_residual[1], lambda[1]);
		}
		void solve()
		{
			T neg_beta = -1 * beta[2];
			axpy<D, T>(streams[2], b_dimension, &neg_beta, b, lambda[1]);//lambda_hat=-beta*b;
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