#include <thread>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <chrono>
namespace alsm
{


	class client
	{
	public:
		std::atomic_int& ready_thread_count;
		std::atomic_bool& update_recieved;
		std::atomic_bool& work_finished;
		const std::chrono::nanoseconds wait_time;
		void recieve_sync()
		{
			while (!update_recieved.load())
			{
				//std::this_thread::sleep_for(wait_time);
			}
			update_recieved.store(false);
		}
		virtual void compute()
		{
			// do something
		}
		void send_sync()
		{
			ready_thread_count++;
		}
		virtual void recieve() = 0;
		virtual void send() = 0;
	public:
		int index;
		client(std::atomic_bool& in_work_finished, std::atomic_bool& in_update_recieved, std::atomic_int& in_free_thread_count,int in_wait_time,int in_index)
			:work_finished(in_work_finished), update_recieved(in_update_recieved), ready_thread_count(in_free_thread_count),wait_time(in_wait_time), index(in_index)
		{

		}
		virtual void work()
		{
			recieve_sync();
			while (!work_finished.load())
			{
				recieve();
				compute();
				send();
				send_sync();
				recieve_sync();
			}
			send_sync();
		}
	};
	class server
	{
	private:
		std::atomic_int& ready_thread_count;
		std::atomic_bool* update_recieved_vector;
		
		const std::chrono::nanoseconds wait_time;
	public:
		std::atomic_bool& work_finished;
		const int client_number;
		const int max_iter;
		int current_iter;
	public:
		void send_sync()
		{
			for (int i = 0; i < client_number; i++)
			{
				update_recieved_vector[i].store(true);
			}
		}
		virtual void compute() = 0;
		void recieve_sync()
		{
			while (ready_thread_count.load() != client_number)
			{
				//std::this_thread::sleep_for(wait_time);
			}
			ready_thread_count.store(0);
		}
		virtual void send() = 0;
		virtual void recieve() = 0;
	public:
		server(std::atomic_int& in_free_thread_count, std::atomic_bool* in_update_recieved_vector, std::atomic_bool& in_work_finished, int in_client_member, int in_wait_time, int in_max_iter)
			:ready_thread_count(in_free_thread_count), update_recieved_vector(in_update_recieved_vector), work_finished(in_work_finished), client_number(in_client_member), wait_time(in_wait_time), max_iter(in_max_iter)
		{
			current_iter = 0;
		}
		virtual void work()
		{
			while (!work_finished.load())
			{
				send();
				send_sync();
				recieve_sync();
				recieve();
				compute();
				current_iter++;
				if (current_iter == max_iter)
				{
					fprintf(stdout, " max iteration %d is exceed\n", max_iter);
					work_finished.store(true);
				}
			}
			send_sync();
			recieve_sync();
		}
	};
}