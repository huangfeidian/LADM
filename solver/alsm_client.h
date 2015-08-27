#ifndef __H_ALSM_CLIENT_H__
#define __H_ALSM_CLIENT_H__
#include "client.h"
#include "../blas/level_1.h"
#include "../blas/level_2.h"
#include "../proximal/proximal.h"
#include <vector>
#include <sstream>
#define FILE_DEBUG 0
namespace alsm
{
	template<DeviceType D, typename T>
	class alsm_client :public client
	{
	public:
		T *A;// we assume the A is compact stored the so lda is either m or n
		MatrixMemOrd A_ord;
		int lda;
		T *x_1, *x_2;
		T* xG;
		bool IdentityMatrix;
		T *lambda_hat, *residual;// residual=Ax
		T* v;//A^T \lambda /\sigma
		T eta;// the paper says eta must be bigger than n*norm{A} so we assign eta=n* norm{A}*2 
		T  sigma;//sigma=eta*beta
		T* client_beta;
		const int x_dimension, b_dimension;
		T *server_x_diff_nrm2;
		T* server_diff_xG_nrm2;
		T *server_opt, *server_residual;
		T* server_x_nrm2;
		stream<D> client_stream;
		const FunctionObj<T> func;
		int server_device_index;
		StopCriteria stop_type;
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

		}
		virtual void send()
		{
			to_server<D, T>(client_stream, server_residual, residual, b_dimension, server_device_index);
			client_stream.sync();
		}
		virtual void compute()
		{
			sigma = eta*(*client_beta);
			//output(x_1);
			if (IdentityMatrix)
			{
				copy<D, T>(client_stream, b_dimension, lambda_hat, v);
			}
			else
			{
				gemv<D, T>(client_stream, MatrixTrans::TRANSPOSE, A_ord, b_dimension, x_dimension, 1, A, lda, lambda_hat,0, v);//v=A^T*lambda_hat
			}
			//output(v);
			axpby<D, T>(client_stream, x_dimension, 1, x_1, -1 / sigma, v);//v=x_1+A^T*lambda_hat/sigma
			//output(v);
			//std::cout << "the sigma is " << sigma << std::endl;

			BatchProxEval<D, T>(client_stream, func, x_dimension, sigma/2, v, x_2);//x_2=prox{func+sigma/2*{x-v}^2}
			//output(x_2);

			nrm2<D, T>(client_stream, x_dimension, x_2, server_x_nrm2);

			axpy<D, T>(client_stream, x_dimension, -1, x_2, x_1);//x_1=x_1-x_2;
			//output(x_1);
			nrm2<D, T>(client_stream, x_dimension, x_1, server_x_diff_nrm2);


			//std::cout <<"eta_norm " <<eta_norm << std::endl;
			//std::swap(x_1, x_2);//x_1=x_2;
			copy<D, T>(client_stream, x_dimension, x_2, x_1);
			if (stop_type==StopCriteria::ground_truth)
			{
				assert(xG != nullptr);
				axpy<D, T>(client_stream, x_dimension, -1.0, xG, x_1);
				nrm2<D, T>(client_stream, x_dimension, x_1, server_diff_xG_nrm2);
				copy<D, T>(client_stream, x_dimension, x_2, x_1);
			}
			//output(x_1);
			if (IdentityMatrix)
			{
				copy<D,T>(client_stream, b_dimension, x_1, residual);
			}
			else
			{
				gemv<D, T>(client_stream, MatrixTrans::NORMAL, A_ord, b_dimension, x_dimension,1, A, lda, x_1, 0, residual);//residual=A*x_1
			}
			if (func.h == UnaryFunc::Abs)
			{
				asum<D, T>(client_stream, x_dimension, x_1,server_opt);
			}
			else
			{
				if (func.h == UnaryFunc::Square)
				{
					nrm2<D, T>(client_stream, x_dimension, x_1, server_opt);
				}
				else
				{
					BatchFuncEval<D, T>(client_stream, func, x_dimension, x_1,server_opt);//eval{func{x_1}}
				}
			}

			//std::cout << opt_value << std::endl;
			
		}
		virtual void task()
		{
			client_stream.set_context();
			recieve();
			compute();
			send();
		}
	public:
		alsm_client(std::atomic_bool* in_work_finished,cache_align_storage<std::atomic_bool>* in_my_turn, std::atomic_int* in_free_thread_count, int in_wait_time, int in_index, int in_b_dimension, int in_x_dimension,
			FunctionObj<T> in_func, stream<D> in_stream)
			:client(in_work_finished, in_my_turn,in_free_thread_count, in_wait_time, in_index), b_dimension(in_b_dimension), x_dimension(in_x_dimension), func(in_func), client_stream(in_stream)
		{

		}

		void init_problem(bool is_identity, MatrixMemOrd in_A_ord, T* in_A, T* in_x, T* in_v, T* in_client_beta, T* in_client_lambda, T* in_client_residual, T in_eta,StopCriteria in_stop_type,T* in_xG=nullptr)
		{
			IdentityMatrix = is_identity;
			A_ord = in_A_ord;
			lda = (A_ord == MatrixMemOrd::ROW) ? x_dimension : b_dimension;
			x_1 = in_x;
			x_2 = x_1 + x_dimension;
			lambda_hat = in_client_lambda;
			A = in_A;
			v = in_v;
			xG = in_xG;
			client_beta = in_client_beta;
			residual = in_client_residual;
			eta = in_eta;
			stop_type = in_stop_type;
			if ((stop_type == StopCriteria::ground_truth||stop_type==StopCriteria::ground_object)&& in_xG == nullptr)
			{
				printf("the stop criteria is ground truth but the xG is null\n");
				exit(1);
			}
			//std::cout << "the eta is " << eta << std::endl;
#if FILE_DEBUG
			std::stringstream file_name;
			file_name << "x_debug" << index << ".csv";
			std::string temp;
			file_name >> temp;
			x_debug = fopen(temp.c_str(), "wb");
			//printf("a:%lf'\tb:%lf\tc:%lf\td:%lf\te:%lf\n", func.a, func.b, func.c, func.d, func.e);
#endif

		}
		void connect_server(int in_server_device_index, T* in_opt, T* in_residual, T* in_x_old_nrm, T* in_x_diff_nrm, T* in_xG_diff_norm = nullptr)
		{
			server_device_index = in_server_device_index;
			server_x_diff_nrm2 = in_x_diff_nrm;
			server_opt = in_opt;
			server_x_nrm2 = in_x_old_nrm;
			server_residual = in_residual;
			server_diff_xG_nrm2 = in_xG_diff_norm;
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