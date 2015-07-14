#ifndef __H_alsmsolver_H__
#define __H_alsmsolver_H__
#include "solver.h"
#include "../blas/level_1.h"
#include "../blas/level_2.h"
#include "../util/alloca.h"
#include "../proximal/proximal.h"
#include <vector>
#include <sstream>
#define FILE_DEBUG 1
namespace alsm
{
	template<DeviceType D, typename T>
	class alsm_server :public server
	{
	private:
		T *lambda, *lambda_hat;
		T *client_lambda, *client_beta;
		T *b;
		T* beta;
		T rho, beta_max;
		T* client_residual;//Ax_i
		T* total_residual;//sum{Ax_i}-b
		T* eta_norm;//stand for \sqrt{\eta _i}||x_i^{k+1}-x_i^k||
		T* client_opt_value;
		T current_opt_value;//sum{client_opt_value}
		T norm_b;//norm{b}
		T epsilon_1;		//eps_1=norm{total_residual}/norm{b} 
		T epsilon_2;		//eps_2=beta_k* max{eta_norm}
		T current_eps1, current_eps2;
		int b_dimension;
		stream<D> server_stream;
#if FILE_DEBUG
		FILE* scalar_info;
		//std::vector<FILE*> residual_info;
		//FILE* lambda_info;//for lambda het
		//FILE* total_residual_info;
#endif
	public:
		inline void gather_residual()
		{
			T alpha = static_cast<T>(1);
			alsm_memset<D, T>(total_residual, 0, b_dimension);
			for (int i = 0; i < client_number; i++)
			{

				axpy<D, T>(server_stream, b_dimension, &alpha, client_residual + i*b_dimension, total_residual);
			}
			alpha = static_cast<T>(-1);
			axpy<D, T>(server_stream, b_dimension, &alpha, b, total_residual);

		}
		inline void update_lambda_hat()
		{
			copy<D, T>(server_stream, b_dimension, lambda, lambda_hat);
			axpy<D, T>(server_stream, b_dimension, beta, total_residual, lambda_hat);
		}
		inline void update_lambda()
		{
			axpy<D, T>(server_stream, b_dimension, beta, total_residual, lambda);
		}
		void update_beta()
		{
			T max_eta_norm = static_cast<T>(0);

			for (int i = 0; i < client_number; i++)
			{
#if FILE_DEBUG
				fprintf(scalar_info, "%lf,", static_cast<double>(eta_norm[i]));
#endif
				if (eta_norm[i]>max_eta_norm)
				{
					max_eta_norm = eta_norm[i];

				}
			}

			//std::cout << "eta norm " << max_eta_norm<<std::endl;
			current_eps2 = *beta*max_eta_norm / norm_b;
			T total_residual_norm = 0;
			nrm2<D, T>(server_stream, b_dimension, total_residual, &total_residual_norm);
			current_eps1 = total_residual_norm / norm_b;
#if FILE_DEBUG
			fprintf(scalar_info, "%lf,%lf,", static_cast<double>(current_eps1), static_cast<double>(current_eps2));
#endif
			current_opt_value = 0;
			for (int i = 0; i < client_number; i++)
			{
				current_opt_value += client_opt_value[i];
#if FILE_DEBUG
				fprintf(scalar_info, "%lf,", static_cast<double>(client_opt_value[i]));
#endif
				//std::cout << client_opt_value[i]<<"\t";
			}
			//std::cout << std::endl;
#if FILE_DEBUG
			fprintf(scalar_info, "%lf,%lf,%lf,", static_cast<double>(max_eta_norm), static_cast<double>(total_residual_norm), static_cast<double>(current_opt_value));
#endif

			if (current_eps2< epsilon_2)
			{
				*beta = *beta*rho;
				if (current_eps1< epsilon_1)
				{
					work_finished.store(true);
				}
			}
			if (*beta > beta_max)
			{
				*beta = beta_max;
			}
			
#if FILE_DEBUG
			fprintf(scalar_info, "%lf\n", static_cast<double>(*beta));
#endif
		}
		virtual void compute()
		{
			gather_residual();
			update_lambda();
			update_beta();
			update_lambda_hat();
			//output_train_result();
		}

		void output_train_result()
		{
			current_opt_value = static_cast<T>(0);
			std::cout << "opt£º ";
			for (int i = 0; i < client_number; i++)
			{
				current_opt_value += client_opt_value[i];
			}
			std::cout <<current_opt_value<< std::endl;
			//printf("beta %f eps1 %f eps2 %f   opt%f at %4d iter \n",static_cast<double>(*beta), static_cast<double>(current_eps1), static_cast<double>(current_eps2),
			//	static_cast<double>(current_opt_value), current_iter);
		}
		virtual void send()
		{
			for (int i = 0; i < client_number; i++)
			{
				client_beta[i] = *beta;
			}
		}
		virtual void recieve()
		{
			
		}
	public:
		alsm_server(std::atomic_int& in_free_thread_count, std::atomic_bool* in_update_recieved_vector,
			std::atomic_bool& in_work_finished, int in_client_number, int in_wait_time, int in_max_iter, int in_b_dimesion, stream<D>& in_stream) :
			server(in_free_thread_count, in_update_recieved_vector, in_work_finished, in_client_number, in_wait_time, in_max_iter), b_dimension(in_b_dimesion), server_stream(in_stream)
		{


		}
		void init_problem(T* in_client_residual, T* in_client_opt, T* in_client_eta_norm, T* in_b, T* in_beta, T* in_client_beta, T* in_lambda_hat, T* in_lambda, T* in_total_residual)
		{

			b = in_b;
			beta = in_beta;
			lambda_hat = in_lambda_hat;
			lambda = in_lambda;
			total_residual = in_total_residual;
			client_residual = in_client_residual;
			client_opt_value = in_client_opt;
			client_beta = in_client_beta;
			eta_norm = in_client_eta_norm;
			nrm2<D, T>(server_stream, b_dimension, b, &norm_b);
#if FILE_DEBUG
			set_debug_file();
#endif
			//std::cout << "norm b " << norm_b << std::endl;
		}
		void init_parameter(T in_eps1, T in_eps2, T in_beta,T in_beta_max, T in_rho)
		{
			rho = in_rho;
			*beta = in_beta;
			beta_max = in_beta_max;
			epsilon_1 = in_eps1;
			epsilon_2 = in_eps2;
		}
		void set_debug_file()
		{
#if FILE_DEBUG
			scalar_info =fopen("scalar_info.csv","wb");
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
	template<DeviceType D, typename T>
	class alsm_client :public client
	{
	private:
		T *A;// we assume the A is compact stored the so lda is either m or n
		MatrixMemOrd A_ord;
		int lda;
		T *x_1, *x_2;
		bool IdentityMatrix;
		T *lambda_hat, *residual;// residual=Ax
		T* v;//A^T \lambda /\sigma
		T eta;// the paper says eta must be bigger than n*norm{A} so we assign eta=n* norm{A}*2 
		T  sigma, opt_value;//sigma=eta*beta
		T* client_beta;
		const int x_dimension, b_dimension;
		T eta_norm;//eta norm=\sqrt{eta}*norm{x^{k+1}-x^{k}}
		T *server_eta_nrm, *server_opt, *server_residual, *server_lambda_hat ;
		stream<D>& client_stream;
		const FunctionObj<T> func;
		
#if FILE_DEBUG
		FILE* x_debug;
#endif
	public:
		void output(T* in_x)
		{
#if FILE_DEBUG
			//for (int i = 0; i < 10; i++)
			//{
			//	fprintf(x_debug, "%lf,", static_cast<double>(in_x[i]));
			//}
			//fprintf(x_debug, "%lf\n", 0);
#endif
		}
		virtual void recieve()
		{
			copy<D, T>(client_stream, b_dimension, server_lambda_hat,lambda_hat );
			sigma = eta*(*client_beta);
		}
		virtual void send()
		{
			*server_eta_nrm = eta_norm;
			*server_opt = opt_value;
			copy<D, T>(client_stream, b_dimension, residual, server_residual);
		}
		virtual void  proximal_update()
		{
			T temp_lambda = 1 / sigma;
			//BatchProxEval<D, T>(client_stream, func, x_dimension, sigma, v, x_2);//x_2=prox{func+sigma/2*{x-v}^2}
			for (int i = 0; i < x_dimension; i++)
			{
				if (fabsf(v[i]) <= temp_lambda)
				{
					x_2[i] = 0;
				}
				else
				{
					if (v[i]>0)
					{
						x_2[i] = v[i] - temp_lambda;
					}
					else
					{
						x_2[i] = v[i] + temp_lambda;
					}
				}
			}
		}
		virtual void compute()
		{
			output(x_1);
			T alpha_one = static_cast<T>(1);
			T alpha_neg_one = static_cast<T>(-1);
			T beta_zero = static_cast<T>(0);
			if (IdentityMatrix)
			{
				copy<D, T>(client_stream, b_dimension, lambda_hat, v);
			}
			else
			{
				gemv<D, T>(client_stream, MatrixTrans::TRANSPOSE, A_ord, b_dimension, x_dimension, &alpha_one, A, lda, lambda_hat, &beta_zero, v);//v=A^T*lambda_hat
			}
			//output(v);
			T inver_sigma = -1 / sigma;
			axpby<D, T>(client_stream, x_dimension, &alpha_one, x_1, &inver_sigma, v);//v=x_1+A^T*lambda_hat/sigma
			//output(v);
			sigma /= 2;
			//std::cout << "the sigma is " << sigma << std::endl;

			BatchProxEval<D, T>(client_stream, func, x_dimension, sigma, v, x_2);//x_2=prox{func+sigma/2*{x-v}^2}
			//output(x_2);
			axpy<D, T>(client_stream, x_dimension, &alpha_neg_one, x_2, x_1);//x_1=x_1-x_2;
			//output(x_1);
			nrm2<D, T>(client_stream, x_dimension, x_1, &eta_norm);//eta_norm=nrm{x_1-x_2}
			eta_norm *= sqrt(eta);//eta_norm=nrm{x_1-x_2}*\sqrt{eta}
			//std::cout <<"eta_norm " <<eta_norm << std::endl;
			std::swap(x_1, x_2);//x_1=x_2;
			//output(x_1);
			client_stream.sync();
			if (IdentityMatrix)
			{
				copy(client_stream, b_dimension, x_1, residual);
			}
			else
			{
				gemv<D, T>(client_stream, MatrixTrans::NORMAL, A_ord, b_dimension, x_dimension, &alpha_one, A, lda, x_1, &beta_zero, residual);//residual=A*x_1
			}
			if (func.h == UnaryFunc::Abs)
			{
				asum<D, T>(client_stream, x_dimension, x_1, &opt_value);
			}
			else
			{
				if (func.h == UnaryFunc::Square)
				{
					nrm2<D, T>(client_stream, x_dimension, x_1, &opt_value);
				}
				else
				{
					BatchFuncEval<D, T>(client_stream, func, x_dimension, x_1, &opt_value);//eval{func{x_1}}
				}
			}
			
			//std::cout << opt_value << std::endl;
			client_stream.sync();
		}
	public:
		alsm_client(std::atomic_bool& in_work_finished, std::atomic_bool& in_update_recieved, std::atomic_int& in_free_thread_count,int in_wait_time,int in_index, int in_b_dimension, int in_x_dimension,
			FunctionObj<T> in_func, stream<D>& in_stream)
			:client(in_work_finished, in_update_recieved, in_free_thread_count,in_wait_time,in_index), b_dimension(in_b_dimension), x_dimension(in_x_dimension), func(in_func), client_stream(in_stream)
		{

		}
		void init_problem(bool is_identity,MatrixMemOrd in_A_ord,T* in_A, T* in_x, T* in_v, T* in_client_beta,T* in_client_lambda,T* in_client_residual,T in_eta)
		{
			IdentityMatrix = is_identity;
			A_ord = in_A_ord;
			lda = (A_ord == MatrixMemOrd::ROW) ? x_dimension: b_dimension;
			x_1 = in_x;
			x_2 = x_1+x_dimension;
			lambda_hat = in_client_lambda;
			A = in_A;
			v = in_v;
			client_beta = in_client_beta;
			residual = in_client_residual;
			eta = in_eta;
			std::cout << "the eta is " << eta << std::endl;
#if FILE_DEBUG
			std::stringstream file_name;
			file_name << "x_debug" << index << ".csv";
			std::string temp;
			file_name >> temp;
			x_debug = fopen(temp.c_str(), "wb");
			//printf("a:%lf'\tb:%lf\tc:%lf\td:%lf\te:%lf\n", func.a, func.b, func.c, func.d, func.e);
#endif
			
		}
		void connect_server(T* in_eta_nrm, T* in_opt, T* in_residual, T* in_lambda_hat)
		{
			server_eta_nrm = in_eta_nrm;
			server_opt = in_opt;
			server_residual = in_residual;
			server_lambda_hat = in_lambda_hat;
		}
		~alsm_client()
		{
#if FILE_DEBUG
			//fclose(x_debug);
#endif
		}
	};
}
#endif
