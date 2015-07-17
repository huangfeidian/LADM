#ifndef __H_SERVER_H__
#define __H_SERVER_H__
#include <thread>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <array>
namespace alsm
{
	class server
	{
	public:
		std::atomic_int& ready_thread_count;
		std::atomic_int* client_turns;
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
				client_turns[i].store(1);
			}
			//printf("server update send\n");
		}
		virtual void compute() = 0;
		void recieve_sync()
		{
			while (ready_thread_count.load() != client_number)
			{
				//std::this_thread::sleep_for(wait_time);
			}
			//printf("server update recieved\n");
			ready_thread_count.store(0);
		}
		virtual void send() = 0;
		virtual void recieve() = 0;
	public:
		server(std::atomic_int& in_free_thread_count, std::atomic_int* in_client_turns,std::atomic_bool& in_work_finished, int in_client_member, int in_wait_time, int in_max_iter)
			:ready_thread_count(in_free_thread_count), client_turns(in_client_turns), work_finished(in_work_finished), client_number(in_client_member), wait_time(in_wait_time), max_iter(in_max_iter)
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
#endif