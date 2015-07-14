#ifndef __H_proxeval_H__
#define __H_proxeval_H__

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "unary_function.h"
namespace alsm
{
	namespace 
	{
		// Proximal operator definitions.
		//
		// Each of the following functions corresponds to one of the Function enums.
		// All functions accept one argument x and five parameters (a, b, c, d and rho)
		// and returns the evaluation of
		//
		//   x -> Prox{c * f(a * x - b) + d * x + e * x ^ 2},
		//
		// where Prox{.} is the proximal operator with penalty parameter rho.
		using namespace functions;
		template <typename T>
		__DEVICE__  T ProxAbs(T v, T rho)
		{
			T a = MaxPos(v - 1/rho);
			T b = MaxNeg(v + 1 / rho);
			//std::cout << a << "," << b << ","<<a-b<<" ";
			return a - b;
		}

		template <typename T>
		__DEVICE__  T ProxNegEntr(T v, T rho)
		{
			// Use double precision.
			return static_cast<T>(
				LambertWExp<double>(
				static_cast<double>((rho * v - 1) + Log(rho)))) / rho;
		}

		template <typename T>
		__DEVICE__  T ProxExp(T v, T rho)
		{
			return v - static_cast<T>(
				LambertWExp<double>(static_cast<double>(v - Log(rho))));
		}

		template <typename T>
		__DEVICE__  T ProxHuber(T v, T rho)
		{
			return Abs(v) < 1 + 1 / rho ? v * rho / (1 + rho) : v - Sign(v) / rho;
		}

		template <typename T>
		__DEVICE__  T ProxIdentity(T v, T rho)
		{
			return v - 1 / rho;
		}

		template <typename T>
		__DEVICE__  T ProxIndBox01(T v, T rho)
		{
			return v <= 0 ? 0 : v >= 1 ? 1 : v;
		}

		template <typename T>
		__DEVICE__  T ProxIndEq0(T v, T rho)
		{
			return 0;
		}

		template <typename T>
		__DEVICE__  T ProxIndGe0(T v, T rho)
		{
			return v <= 0 ? 0 : v;
		}

		template <typename T>
		__DEVICE__  T ProxIndLe0(T v, T rho)
		{
			return v >= 0 ? 0 : v;
		}

		template <typename T>
		__DEVICE__  T ProxLogistic(T v, T rho)
		{
			// Initial guess based on piecewise approximation.
			T x;
			if (v < static_cast<T>(-2.5))
				x = v;
			else if (v > static_cast<T>(2.5) + 1 / rho)
				x = v - 1 / rho;
			else
				x = (rho * v - static_cast<T>(0.5)) / (static_cast<T>(0.2) + rho);

			// Newton iteration.
			T l = v - 1 / rho, u = v;
			for (unsigned int i = 0; i < 5; ++i)
			{
				T inv_ex = 1 / (1 + Exp(-x));
				T f = inv_ex + rho * (x - v);
				T g = inv_ex * (1 - inv_ex) + rho;
				if (f < 0)
					l = x;
				else
					u = x;
				x = x - f / g;
				x = Min(x, u);
				x = Max(x, l);
			}

			// Guarded method if not converged.
			for (unsigned int i = 0; u - l > Tol<T>() && i < 100; ++i)
			{
				T g_rho = 1 / (rho * (1 + Exp(-x))) + (x - v);
				if (g_rho > 0)
				{
					l = Max(l, x - g_rho);
					u = x;
				}
				else
				{
					u = Min(u, x - g_rho);
					l = x;
				}
				x = (u + l) / 2;
			}
			return x;
		}

		template <typename T>
		__DEVICE__  T ProxMaxNeg0(T v, T rho)
		{
			T z = v >= 0 ? v : 0;
			return v + 1 / rho <= 0 ? v + 1 / rho : z;
		}

		template <typename T>
		__DEVICE__  T ProxMaxPos0(T v, T rho)
		{
			T z = v <= 0 ? v : 0;
			return v >= 1 / rho ? v - 1 / rho : z;
		}

		template <typename T>
		__DEVICE__  T ProxNegLog(T v, T rho)
		{
			return (v + Sqrt(v * v + 4 / rho)) / 2;
		}

		template <typename T>
		__DEVICE__  T ProxRecipr(T v, T rho)
		{
			v = Max(v, static_cast<T>(0));
			return CubicSolve(-v, static_cast<T>(0), -1 / rho);
		}

		template <typename T>
		__DEVICE__  T ProxSquare(T v, T rho)
		{
			return rho * v / (1 + rho);
		}

		template <typename T>
		__DEVICE__  T ProxZero(T v, T rho)
		{
			return v;
		}
	}
	// Evaluates the proximal operator of f.
	template <typename T>
	__DEVICE__ T ProxEval(const  FunctionObj<T> &f_obj, T v, T rho)
	{
		const T a = f_obj.a, b = f_obj.b, c = f_obj.c, d = f_obj.d, e = f_obj.e;
		v = a * (v * rho - d) / (e + rho) - b;
		rho = (e + rho) / (c * a * a);
		switch (f_obj.h)
		{
		case UnaryFunc::Abs: v = ProxAbs(v, rho); break;
		case UnaryFunc::NegEntr: v = ProxNegEntr(v, rho); break;
		case UnaryFunc::Exp: v = ProxExp(v, rho); break;
		case UnaryFunc::Huber: v = ProxHuber(v, rho); break;
		case UnaryFunc::Identity: v = ProxIdentity(v, rho); break;
		case UnaryFunc::IndBox01: v = ProxIndBox01(v, rho); break;
		case UnaryFunc::IndEq0: v = ProxIndEq0(v, rho); break;
		case UnaryFunc::IndGe0: v = ProxIndGe0(v, rho); break;
		case UnaryFunc::IndLe0: v = ProxIndLe0(v, rho); break;
		case UnaryFunc::Logistic: v = ProxLogistic(v, rho); break;
		case UnaryFunc::MaxNeg0: v = ProxMaxNeg0(v, rho); break;
		case UnaryFunc::MaxPos0: v = ProxMaxPos0(v, rho); break;
		case UnaryFunc::NegLog: v = ProxNegLog(v, rho); break;
		case UnaryFunc::Recipr: v = ProxRecipr(v, rho); break;
		case UnaryFunc::Square: v = ProxSquare(v, rho); break;
		case UnaryFunc::Zero: default: v = ProxZero(v, rho); break;
		}
		//std::cout << b << "," << a << "," << (v + b) / a << "\t";
		return (v + b) / a;
	}
}
#endif
