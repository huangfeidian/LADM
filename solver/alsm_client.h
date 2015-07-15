#ifndef __H_ALSM_CLIENT_H__
#define __H_ALSM_CLIENT_H__
#include "client.h"
#include "../blas/level_1.h"
#include "../blas/level_2.h"
#include "../proximal/proximal.h"
#include <vector>
#include <sstream>
#define FILE_DEBUG 1
namespace alsm
{
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
		T *server_eta_nrm, *server_opt, *server_residual, *server_lambda_hat;
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
			copy<D, T>(client_stream, b_dimension, server_lambda_hat, lambda_hat);
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
			//std::swap(x_1, x_2);//x_1=x_2;
			copy<D, T>(client_stream, x_dimension, x_2, x_1);
			//output(x_1);
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
		alsm_client(std::atomic_bool& in_work_finished,std::atomic_int& in_my_turn, std::atomic_int& in_free_thread_count, int in_wait_time, int in_index, int in_b_dimension, int in_x_dimension,
			FunctionObj<T> in_func, stream<D>& in_stream)
			:client(in_work_finished, in_my_turn,in_free_thread_count, in_wait_time, in_index), b_dimension(in_b_dimension), x_dimension(in_x_dimension), func(in_func), client_stream(in_stream)
		{

		}
		void init_problem(bool is_identity, MatrixMemOrd in_A_ord, T* in_A, T* in_x, T* in_v, T* in_client_beta, T* in_client_lambda, T* in_client_residual, T in_eta)
		{
			IdentityMatrix = is_identity;
			A_ord = in_A_ord;
			lda = (A_ord == MatrixMemOrd::ROW) ? x_dimension : b_dimension;
			x_1 = in_x;
			x_2 = x_1 + x_dimension;
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