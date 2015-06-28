#include "solver/solver.h"
#include "blas/level_1.cu"
#include "blas/level_2.cu"
namespace alsm
{
	template<DeviceType D,typename T>
	class sample_client:public client
	{
	public:
		T* scal_rho;
		T* local_vec;
		T* host_vec;
		stream<D> local_stream;
		int size;
		T local_result;
		T* host_result;
		sample_client(std::atomic_bool& in_work_finished, std::atomic_bool& in_update_recieved, std::atomic_int& in_free_thread_count,T* in_local_vec,T* in_host_vec,T* in_rho,stream<D>& in_stream,int in_size)
			:client(in_work_finished, in_update_recieved, in_free_thread_count), scal_rho(in_rho), local_stream(in_stream), local_vec(in_local_vec), size(in_size)
		{

		}
		virtual void recieve()
		{
			while (update_recieved.load())
			{
				// busy loop
			}
			update_recieved.store(false);
		}
		virtual void compute()
		{
			scal<D, T>(local_stream, size, local_vec, scal_rho);
			nrm2<D, T>(local_stream, size, local_vec, &local_result);

		}
		virtual void send()
		{
			ready_thread_count++;
			*host_result = local_result;
		}
	public:
		void work()
		{
			recieve();
			while (!work_finished.load())
			{
				compute();
				send();
				recieve();
			}
			send_to_host<D, T>(local_stream, size, local_vec, host_vec);
			send();
		}
	};
	template<typename T>
	class sample_server :public server
	{
	public:
		T* client_rho;
		T* server_vec;
		const int total_size;
		T* host_result;
		T total_result;
		const int max_iter;
		int current_iter;
		const T limit;
		T rho;
		stream<DeviceType::CPU> local_stream;
		sample_server(std::atomic_int& in_free_thread_count, std::atomic_bool* in_update_recieved_vector, std::atomic_bool& in_work_finished, 
			int in_client_member, int in_wait_time, T* in_client_rho, T* result, int in_iter, int in_size,T in_limit)
			:server(in_free_thread_count, in_update_recieved_vector, in_work_finished, in_client_member, in_wait_time), client_rho(in_client_rho), host_result(result), max_iter(in_iter), total_size(in_size), limit(in_limit)
		{
			total_result = 0;
			rho = 0;
		}
		virtual void send()
		{
			for (int i = 0; i < client_number; i++)
			{
				client_rho[i] = rho;
				update_recieved_vector[i].store(true);
			}
		}
		virtual void compute()
		{
			total_result = 0;
			for (int i = 0; i < client_number; i++)
			{
				total_result += host_result[i];
			}
			if (total_result < limit)
			{
				std::cout << " the limit " << limit << " is exceed by " << total_result << std::endl;
				work_finished.store(true);
			}
			rho = 1 / sqrt(total_result);
			current_iter++;
			if (current_iter > max_iter)
			{
				std::cout << "the iteration limit " << max_iter << " is exceeded" << std::endl;
				work_finished.store(true);
			}
		}
		virtual void recieve()
		{
			while (ready_thread_count != client_number)
			{
				std::this_thread::sleep_for(wait_time);
			}
			ready_thread_count.store(0);
		}
		void work()
		{
			
			while (work_finished.load())
			{
				send();
				recieve();
				compute();
			}
			send();
			recieve();
			nrm2<DeviceType::CPU, T>(local_stream, total_size, server_vec, &total_result);
			std::cout << "the total norm2 is " << total_result << std::endl;
		}
	};
}