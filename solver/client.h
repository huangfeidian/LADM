#ifndef _H_CLIENT_H_
#define _H_CLIENT_H_
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
		client(std::atomic_bool& in_work_finished, std::atomic_bool& in_update_recieved, std::atomic_int& in_free_thread_count, int in_wait_time, int in_index)
			:work_finished(in_work_finished), update_recieved(in_update_recieved), ready_thread_count(in_free_thread_count), wait_time(in_wait_time), index(in_index)
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