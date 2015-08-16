#ifndef __H_ALSM_SERVER_H__
#define __H_ALSM_SERVER_H__
#include "server.h"
#include "../blas/level_1.h"
#include "../util/alloca.h"
#include <vector>
#include <sstream>
#define FILE_DEBUG 1
using std::vector;
namespace alsm
{
	template<DeviceType D, typename T>
	class alsm_server :public server
	{
	public:
		T beta;
		T rho, beta_max;
		T total_opt_value;//sum{client_opt_value}
		T old_opt_value;
		T norm_b;//norm{b}
		T epsilon_1;		//eps_1=norm{total_residual}/norm{b} 
		T epsilon_2;		//eps_2=beta_k* max{eta_norm}
		T epsilon_3;//for additional stop criterion
		T current_eps1, current_eps2,current_eps3;
		int b_dimension;
		T total_residual_norm;
		stream<D> server_stream;
	public:
		T *server_lambda;
		T *server_lambda_hat;
		T *b;
		T *total_residual;//sum{Ax_i}-b
	public:
		
		vector<T*> clients_beta;
		vector<T*> clients_residual;//for all Ax_i
		vector<T> clients_sqrt_eta;
	public://counters to decide whether to stop
		StopCriteria stop_type;
		vector<T*> clients_xG_diff_nrm2;
		vector<T*> clients_opt_value;
		vector<T*> clients_xdiff_nrm2;
		vector<T*> clients_xold_nrm2;
	public:
		//device dependent memory
		vector<T*> devices_lambda;
		vector <stream<D>> clients_stream;
#if FILE_DEBUG
		FILE* scalar_info;
		//std::vector<FILE*> residual_info;
		//FILE* lambda_info;//for lambda het
		//FILE* total_residual_info;
#endif
	public:
		inline void gather_residual()
		{
			alsm_memset<D, T>(total_residual, 0, b_dimension);
			for (int i = 0; i < client_number; i++)
			{
				axpy<D, T>(server_stream, b_dimension, 1, clients_residual[i], total_residual);
			}
			axpy<D, T>(server_stream, b_dimension, -1, b, total_residual);

		}
		inline void update_lambda_hat()
		{
			copy<D, T>(server_stream, b_dimension, server_lambda, server_lambda_hat);
			axpy<D, T>(server_stream, b_dimension, beta, total_residual, server_lambda_hat);
			server_stream.sync();
		}
		inline void update_lambda()
		{
			axpy<D, T>(server_stream, b_dimension, beta, total_residual, server_lambda);
		}
		void update_beta()
		{
			total_residual_norm = 0;
			nrm2<D, T>(server_stream, b_dimension, total_residual, &total_residual_norm);
			server_stream.sync();
			T max_eta_norm = static_cast<T>(0);

			for (int i = 0; i < client_number; i++)
			{
#if FILE_DEBUG
				fprintf(scalar_info, "%lf,", static_cast<double>(*clients_xdiff_nrm2[i]));
#endif
				if (*clients_xdiff_nrm2[i]*clients_sqrt_eta[i]>max_eta_norm)
				{
					max_eta_norm = *clients_xdiff_nrm2[i]*clients_sqrt_eta[i];

				}
			}

			//std::cout << "eta norm " << max_eta_norm<<std::endl;
			current_eps2 = beta*max_eta_norm / norm_b;
			
			current_eps1 = total_residual_norm / norm_b;
#if FILE_DEBUG
			fprintf(scalar_info, "%lf,%lf,", static_cast<double>(current_eps1), static_cast<double>(current_eps2));
#endif
			total_opt_value = 0;
			for (int i = 0; i < client_number; i++)
			{
				total_opt_value += *clients_opt_value[i];
#if FILE_DEBUG
				fprintf(scalar_info, "%lf,", static_cast<double>(*clients_opt_value[i]));
#endif
				//std::cout << client_opt_value[i]<<"\t";
			}
			//std::cout << std::endl;
#if FILE_DEBUG
			fprintf(scalar_info, "%lf,%lf,%lf,", static_cast<double>(max_eta_norm), static_cast<double>(total_residual_norm), static_cast<double>(total_opt_value));
#endif

			if (current_eps2 < epsilon_2)
			{
				beta = beta*rho;
			}
			if (beta > beta_max)
			{
				beta = beta_max;
			}
			switch (stop_type)
			{
			case StopCriteria::ground_truth:
				{
					T total_xG_diff = 0;
					for (auto i : clients_xG_diff_nrm2)
					{
						total_xG_diff += (*i)*(*i);
					}
					total_xG_diff = sqrt(total_xG_diff);
					if (total_xG_diff <= epsilon_3)
					{
						work_finished->store(true);
					}
				}
				
				break;
			case StopCriteria::duality_gap:
				if (current_eps1 < epsilon_1)
				{
					work_finished->store(true);
				}
				break;
			case StopCriteria::dual_tol:
				if (current_eps1 < epsilon_1&&current_eps2 < epsilon_2)
				{
					work_finished->store(true);
				}
				break;
			case StopCriteria::increment:
			{
				T total_old_nrm = 0;
				T total_diff_nrm = 0;
				for (auto i : clients_xdiff_nrm2)
				{
					total_diff_nrm += (*i)*(*i);
				}
				for (auto i : clients_xold_nrm2)
				{
					total_old_nrm += (*i)*(*i);
				}
				total_diff_nrm = sqrt(total_diff_nrm);
				total_old_nrm = sqrt(total_old_nrm);
				if (total_diff_nrm < epsilon_3*total_old_nrm)
				{
					work_finished->store(true);
				}
			}
				break;
			case StopCriteria::objective_value:
				if (abs(old_opt_value - total_opt_value) <= old_opt_value*epsilon_3)
				{
					work_finished->store(true);
				}
				old_opt_value = total_opt_value;
			default:
				break;
			}

#if FILE_DEBUG
			fprintf(scalar_info, "%lf\n", static_cast<double>(beta));
#endif
		}
		virtual void compute()
		{
			gather_residual();
			update_lambda();
			update_beta();
			update_lambda_hat();
		}

		void output_train_result()
		{
			total_opt_value = static_cast<T>(0);
			std::cout << "opt£º ";
			for (int i = 0; i < client_number; i++)
			{
				total_opt_value += *clients_opt_value[i];
			}
			std::cout << total_opt_value << std::endl;
			//printf("beta %f eps1 %f eps2 %f   opt%f at %4d iter \n",static_cast<double>(beta), static_cast<double>(current_eps1), static_cast<double>(current_eps2),
			//	static_cast<double>(current_opt_value), current_iter);
		}
		virtual void send()
		{
			for (int i = 0; i < client_number; i++)
			{
				*clients_beta[i] = beta;
			}
			for (int i = 0; i < client_number; i++)
			{
				from_server<D, T>(server_stream, devices_lambda[i], server_lambda_hat, b_dimension, clients_stream[i].device_index);
			}
			server_stream.sync();
		}
		virtual void recieve()
		{

		}
	public:
		alsm_server(std::atomic_int* in_free_thread_count, cache_align_storage<std::atomic_bool>* in_client_turns, std::atomic_bool* in_work_finished, int in_client_number,
			int in_wait_time, int in_max_iter, int in_b_dimesion, stream<D>& in_stream) :
			server(in_free_thread_count,in_client_turns, in_work_finished, in_client_number, in_wait_time, in_max_iter), 
			b_dimension(in_b_dimesion), server_stream(in_stream), old_opt_value(0)
		{


		}
		void add_client(T* client_opt_value, T* client_beta, T client_eta, T* client_residual, T* client_lambda, T* client_xold_nrm2,T* client_xdiff_nrm2,stream<D> client_stream,T* client_xG_diff=nullptr)
		{
			clients_opt_value.push_back(client_opt_value);
			clients_beta.push_back(client_beta);
			clients_sqrt_eta.push_back(std::sqrt(client_eta));
			clients_xold_nrm2.push_back(client_xold_nrm2);
			clients_xdiff_nrm2.push_back(client_xdiff_nrm2);
			clients_residual.push_back(client_residual);
			devices_lambda.push_back(client_lambda);
			clients_stream.push_back(client_stream);
			clients_xG_diff_nrm2.push_back(client_xG_diff);
		}
		void init_problem( T* in_b,  T* in_lambda_hat, T* in_lambda, T* in_total_residual,StopCriteria in_stop_type=StopCriteria::dual_tol)
		{
			stop_type = in_stop_type;
			b = in_b;
			server_lambda_hat = in_lambda_hat;
			server_lambda = in_lambda;
			total_residual = in_total_residual;
			nrm2<D, T>(server_stream, b_dimension, b, &norm_b);
			server_stream.sync();
#if FILE_DEBUG
			set_debug_file();
#endif
			//std::cout << "norm b " << norm_b << std::endl;
		}
		void init_parameter(T in_eps1, T in_eps2, T in_beta, T in_beta_max, T in_rho,T in_eps3)
		{
			rho = in_rho;
			beta = in_beta;
			beta_max = in_beta_max;
			epsilon_1 = in_eps1;
			epsilon_2 = in_eps2;
			epsilon_3 = in_eps3;
		}
		void set_debug_file()
		{
#if FILE_DEBUG
			scalar_info = fopen("scalar_info.csv", "wb");
			//lambda_info = fopen("lambda_info.csv", "wb");
			//total_residual_info = fopen("server_residual_info.csv", "wb");
			for (int i = 0; i < client_number; i++)
			{
				fprintf(scalar_info, "eta_norm%d,", i);
			}
			fprintf(scalar_info, "eps1,eps2,");
			for (int i = 0; i < client_number; i++)
			{
				fprintf(scalar_info, "client_opt%d,", i);
			}
			fprintf(scalar_info, "max_eta_norm,total_residual_norm,current_opt_value,beta\n");
			//std::stringstream file_name_stream;
			//for (int i = 0; i < client_number; i++)
			//{
			//	file_name_stream << "vector_file_" << i << ".csv" << std::endl;
			//	std::string file_name;
			//	file_name_stream >> file_name;
			//	FILE* temp_file = fopen(file_name.c_str(), "wb");
			//	residual_info.push_back(temp_file);

			//}
#endif
		}
		~alsm_server()
		{
#if FILE_DEBUG
			//fclose(scalar_info);
#endif
		}
	};
}
#endif