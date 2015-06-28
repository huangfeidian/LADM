#ifndef __H_proxsubg_H__
#define __H_proxsubg_H__
#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>
#include "unary_function.h"



namespace alsm
{
	namespace 
	{
		// Projection onto subgradient definitions
		//
		// Each of the following functions corresponds to one of the Function enums.
		// All functions accept one argument x and five parameters (a, b, c, d, and e)
		// and returns the evaluation of
		//
		//   x -> ProjSubgrad{c * f(a * x - b) + d * x + (1/2) e * x ^ 2},
		//
		// where ProjSubgrad{.} is the projection  onto the subgradient of the function.
		using namespace functions;
		template <typename T>
		__DEVICE__  T ProjSubgradAbs(T v, T x)
		{
			if (x < static_cast<T>(0.))
				return static_cast<T>(-1.);
			else if (x > static_cast<T>(0.))
				return static_cast<T>(1.);
			else
				return Max(static_cast<T>(-1.), Min(static_cast<T>(1.), v));
		}

		template <typename T>
		__DEVICE__  T ProjSubgradNegEntr(T v, T x)
		{
			return -Log(x) - static_cast<T>(1.);
		}

		template <typename T>
		__DEVICE__  T ProjSubgradExp(T v, T x)
		{
			return Exp(x);
		}

		template <typename T>
		__DEVICE__  T ProjSubgradHuber(T v, T x)
		{
			return Max(static_cast<T>(-1.), Min(static_cast<T>(1.), x));
		}

		template <typename T>
		__DEVICE__  T ProjSubgradIdentity(T v, T x)
		{
			return static_cast<T>(1.);
		}

		template <typename T>
		__DEVICE__  T ProjSubgradIndBox01(T v, T x)
		{
			if (x <= static_cast<T>(0.))
				return Min(static_cast<T>(0.), v);
			else if (x >= static_cast<T>(1.))
				return Max(static_cast<T>(0.), v);
			else
				return static_cast<T>(0.);
		}

		template <typename T>
		__DEVICE__  T ProjSubgradIndEq0(T v, T x)
		{
			return v;
		}

		template <typename T>
		__DEVICE__  T ProjSubgradIndGe0(T v, T x)
		{
			if (x <= static_cast<T>(0.))
				return Min(static_cast<T>(0.), v);
			else
				return static_cast<T>(0.);
		}

		template <typename T>
		__DEVICE__  T ProjSubgradIndLe0(T v, T x)
		{
			if (x >= static_cast<T>(0.))
				return Max(static_cast<T>(0.), v);
			else
				return static_cast<T>(0.);
		}

		template <typename T>
		__DEVICE__  T ProjSubgradLogistic(T v, T x)
		{
			return Exp(x) / (static_cast<T>(1.) + Exp(x));
		}

		template <typename T>
		__DEVICE__  T ProjSubgradMaxNeg0(T v, T x)
		{
			if (x < static_cast<T>(0.))
				return static_cast<T>(-1.);
			else if (x > static_cast<T>(0.))
				return static_cast<T>(0.);
			else
				return Min(static_cast<T>(0.), Max(static_cast<T>(-1.), v));
		}

		template <typename T>
		__DEVICE__  T ProjSubgradMaxPos0(T v, T x)
		{
			if (x < static_cast<T>(0.))
				return static_cast<T>(0.);
			else if (x > static_cast<T>(0.))
				return static_cast<T>(1.);
			else
				return Min(static_cast<T>(1.), Max(static_cast<T>(0.), v));
		}

		template <typename T>
		__DEVICE__  T ProjSubgradNegLog(T v, T x)
		{
			return static_cast<T>(-1.) / x;
		}

		template <typename T>
		__DEVICE__  T ProjSubgradRecipr(T v, T x)
		{
			return static_cast<T>(1.) / (x * x);
		}

		template <typename T>
		__DEVICE__  T ProjSubgradSquare(T v, T x)
		{
			return x;
		}

		template <typename T>
		__DEVICE__  T ProjSubgradZero(T v, T x)
		{
			return static_cast<T>(0.);
		}
	}

	// Evaluates the projection of v onto the subgradient of f at x.
	template <typename T>
	__DEVICE__ T SubgradEval(const FunctionObj<T> &f_obj, T v, T x)
	{
		const T a = f_obj.a, b = f_obj.b, c = f_obj.c, d = f_obj.d, e = f_obj.e;
		if (a == static_cast<T>(0.) || c == static_cast<T>(0.))
			return d + e * x;
		v = static_cast<T>(1.) / (a * c) * (v - d - e * x);
		T axb = a * x - b;
		switch (f_obj.h)
		{
		case UnaryFunc::Abs: v = ProjSubgradAbs(v, axb); break;
		case UnaryFunc::NegEntr: v = ProjSubgradNegEntr(v, axb); break;
		case UnaryFunc::Exp: v = ProjSubgradExp(v, axb); break;
		case UnaryFunc::Huber: v = ProjSubgradHuber(v, axb); break;
		case UnaryFunc::Identity: v = ProjSubgradIdentity(v, axb); break;
		case UnaryFunc::IndBox01: v = ProjSubgradIndBox01(v, axb); break;
		case UnaryFunc::IndEq0: v = ProjSubgradIndEq0(v, axb); break;
		case UnaryFunc::IndGe0: v = ProjSubgradIndGe0(v, axb); break;
		case UnaryFunc::IndLe0: v = ProjSubgradIndLe0(v, axb); break;
		case UnaryFunc::Logistic: v = ProjSubgradLogistic(v, axb); break;
		case UnaryFunc::MaxNeg0: v = ProjSubgradMaxNeg0(v, axb); break;
		case UnaryFunc::MaxPos0: v = ProjSubgradMaxPos0(v, axb); break;
		case UnaryFunc::NegLog: v = ProjSubgradNegLog(v, axb); break;
		case UnaryFunc::Recipr: v = ProjSubgradRecipr(v, axb); break;
		case UnaryFunc::Square: v = ProjSubgradSquare(v, axb); break;
		case UnaryFunc::Zero: default: v = ProjSubgradZero(v, axb); break;
		}
		return a * c * v + d + e * x;
	}
}
#endif
