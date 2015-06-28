// this l1 solver is to solve 
// minimize |x|+|e|_2^2 subject to Ax+e=b
//min |x|1 + |e|1 s.t. Ax + e = b\n
#include "alsm_solver.h"
#include <array>
namespace alsm
{
	template <DeviceType D,typename T> class l1_solver
	{
	public:
		const int client_number;
	private:
		std::atomic_int ready_thread_count;
		std::array<std::atomic_bool,2> update_recieved_vector;
		std::atomic_bool work_finished;
		const std::chrono::microseconds wait_time;
		int x_dimension, b_dimension;
		std::array<stream<D, T>, 3> streams;
		alsm_client<D, T> e_client;
		alsm_client<D, T> x_client;
		alsm_server<D, T> server;
		
	public:
		const int client_number;
		l1_solver(int in_client_number,std::array<stream<D>,3> three_stream, int in_x_dimension,int in_b_dimension ,int in_max_iter,int in_wait_ms)
			:client_number(in_client_number), ready_thread_count{ 0 }, update_recieved_vector{ std::atomic_bool{ false }, 2 }, work_finished{ false }, x_dimension(in_x_dimension), b_dimension(in_b_dimension), wait_time(in_wait_ms), streams(three_stream),
			e_client(work_finished, update_recieve[0], ready_thread_count, 0, in_b_dimension, in_x_dimension, FunctionObj<T>(UnaryFunc::Square), three_stream[0]),
			x_client(work_finished, update_recieve[1], ready_thread_count, 1, in_b_dimension, in_x_dimension, FunctionObj<T>(UnaryFunc::Abs), three_stream[2]),
			server(ready_thread_count,&update_recieve_vector[0],work_finished,client_number,wait_time,in_max_iter,in_b_dimension,three_stream[2])
		{

		}
	private:
		
		T* client_residual[2];
		T* server_residual;
		
		T* x[3];//x[2]=v_x
		T* e[3];//e[2]=v_e
		T opt[3];
		T eta_norm[3];
		T* lambda[2];//lambda[1] for lambda_hat
		T beta[2];
		T* Identity;
		// the memory below has been allocated before we create the solver
		T* A;
		T* b;
		T* output_x;
		T* output_e;
		void init_memory()
		{
			Identity = alsm_malloc<D, T>(b_dimension, b_dimension);
			alsm_memset<D, T>(Identity, 0, b_dimension* b_dimension);
			for (int i = 0; i < b_dimension; i++)
			{
				Identity[i*b_dimension + i] = 1;
			}
			client_residual[0] = alsm_malloc<D, T>(b_dimension*2);
			alsm_memset<D, T>(client_residual[0], 0, 2 * b_dimension);
			client_residual[1] = client_residual[0] + b_dimension;
			server_residual = alsm_malloc<D, T>(b_dimension);
			alsm_memset<D, T>(server_residual, 0, b_dimension);
			x[0]= alsm_malloc<D, T>(3 * x_dimension);
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
		void init_problem(T* in_A, T* in_b, T* in_output_x, T* in_output_e)
		{
			A = in_A;
			b = in_b;
			output_x = in_output_x;
			output_e = in_output_e;
			e_client.init_problem(Identity, e, e[2], beta[0], lambda[0], client_residual[0]);
			x_client.init_problem(A, x, x[2], beta[1], lambda[0], client_residual[1]);
			server.init_problem(client_residual[0], opt, eta_norm, b, beta[2], , beta, lambda[1], lambda[0], server_residual);
			server.init_parameter(100000 * Epsilon<T>(), 10000 * Epsilon<T>(),100,1.2);
			e_client.connect_server(eta_norm + 0, opt + 0, client_residual[0], lambda[1]);
			x_client.connect_server(eta_norm + 1, opt + 1, client_residual[1], lambda[1]);
		}
		void solve()
		{
			std::vector<std::thread> all_threads;
			all_threads.emplace_back(&alsm_server<D, T>::work, &server);
			all_threads.emplace_back(&alsm_client<D, T>::work, &e_client);
			all_threads.emplace_back(&alsm_client<D, T>::work, &x_client);
			for (auto& i : all_threads)
			{
				i.join();
			}
		}
		~l1_solver()
		{
			alsm_memcpy<D, T>(streams[0], output_e, e[0], b_dimension);
			alsm_memcpy<D, T>(streams[1], output_x, x[0], x_dimension);
			alsm_free_all();
		}
	};
}