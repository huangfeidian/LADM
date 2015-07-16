#ifndef _H_CLIENT_H_
#define _H_CLIENT_H_
#include <thread>
#include <memory>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <stdio.h>
namespace alsm
{


	class client
	{
	public:
		std::atomic_int& ready_thread_count;
		std::atomic_int& my_turn;
		std::atomic_bool& work_finished;
		const std::chrono::nanoseconds wait_time;
		void recieve_sync()
		{
			while (my_turn.load()==0)
			{
				//std::this_thread::sleep_for(wait_time);
			}
			my_turn.store(0);
			//printf("client %d update recieved\n",index);
		}
		virtual void compute()
		{
			// do something
		}
		void send_sync()
		{
			ready_thread_count++;
			//int a = ready_thread_count.load();
			//printf("client %d finished ,current count is %d \n",index,a);
		}
		virtual void recieve() = 0;
		virtual void send() = 0;
	public:
		const int index;
		client(std::atomic_bool& in_work_finished, std::atomic_int& in_my_turn, std::atomic_int& in_free_thread_count, int in_wait_time, int in_index)
			:work_finished(in_work_finished),  my_turn(in_my_turn),ready_thread_count(in_free_thread_count), wait_time(in_wait_time), index(in_index)
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
}
#endif