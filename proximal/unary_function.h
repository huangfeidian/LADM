#ifndef __H_unary_H__
#define __H_unary_H__

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>


#include "../util/enum.h"
#include "../util/flags.h"
#include "../util/util.h"


namespace alsm
{
// List of functions supported by the proximal operator library.


// Object associated with the generic function c * f(a * x - b) + d * x.
// Parameters a and c default to 1, while b and d default to 0.

template <typename T>
	struct FunctionObj
	{
		UnaryFunc h;
		T a, b, c, d, e;

		FunctionObj(UnaryFunc h, T a, T b, T c, T d, T e)
			: h(h), a(a), b(b), c(c), d(d), e(e)
		{
			CheckConsts();
		}
		FunctionObj(UnaryFunc h, T a, T b, T c, T d)
			: h(h), a(a), b(b), c(c), d(d), e(0)
		{
			CheckConsts();
		}
		FunctionObj(UnaryFunc h, T a, T b, T c)
			: h(h), a(a), b(b), c(c), d(0), e(0)
		{
			CheckConsts();
		}
		FunctionObj(UnaryFunc h, T a, T b)
			: h(h), a(a), b(b), c(1), d(0), e(0)
		{
		}
		FunctionObj(UnaryFunc h, T a)
			: h(h), a(a), b(0), c(1), d(0), e(0)
		{
		}
		explicit FunctionObj(UnaryFunc h)
			: h(h), a(1), b(0), c(1), d(0), e(0)
		{
		}
		FunctionObj()
			: h(UnaryFunc::Zero), a(1), b(0), c(1), d(0), e(0)
		{
		}

		void CheckConsts()
		{
			if (c < static_cast<T>(0))
				Printf("WARNING c < 0. Function not convex. Using c = 0");
			if (e < static_cast<T>(0))
				Printf("WARNING e < 0. Function not convex. Using e = 0");
			c = std::max(c, static_cast<T>(0));
			e = std::max(e, static_cast<T>(0));
		}
	};


// Local Functions.

	namespace functions
	{

		//  Evaluate abs(x)
		__DEVICE__  double Abs(double x)
		{
			return fabs(x);
		}
		__DEVICE__  float Abs(float x)
		{
			return fabsf(x);
		}

		//  Evaluate acos(x)
		__DEVICE__  double Acos(double x)
		{
			return acos(x);
		}
		__DEVICE__  float Acos(float x)
		{
			return acosf(x);
		}

		//  Evaluate cos(x)
		__DEVICE__  double Cos(double x)
		{
			return cos(x);
		}
		__DEVICE__  float Cos(float x)
		{
			return cosf(x);
		}

		//  Evaluate e^x
		__DEVICE__  double Exp(double x)
		{
			return exp(x);
		}
		__DEVICE__  float Exp(float x)
		{
			return expf(x);
		}

		//  Evaluate log(x)
		__DEVICE__  double Log(double x)
		{
			return log(x);
		}
		__DEVICE__  float Log(float x)
		{
			return logf(x);
		}

		//  Evaluate max(x, y)
		__DEVICE__  double Max(double x, double y)
		{
			return fmax(x, y);
		}
		__DEVICE__  float Max(float x, float y)
		{
			return fmaxf(x, y);
		}

		//  Evaluate max(x, y)
		__DEVICE__  double Min(double x, double y)
		{
			return fmin(x, y);
		}
		__DEVICE__  float Min(float x, float y)
		{
			return fminf(x, y);
		}

		//  Evaluate x^y
		__DEVICE__  double Pow(double x, double y)
		{
			return pow(x, y);
		}
		__DEVICE__  float Pow(float x, float y)
		{
			return powf(x, y);
		}

		//  Evaluate sqrt(x)
		__DEVICE__  double Sqrt(double x)
		{
			return sqrt(x);
		}
		__DEVICE__  float Sqrt(float x)
		{
			return sqrtf(x);
		}

		// Numeric Epsilon.
		template <typename T>
		__DEVICE__  T Epsilon();
		template <>
		__DEVICE__  double Epsilon<double>()
		{
			return 4e-16;
		}
		template <>
		__DEVICE__  float Epsilon<float>()
		{
			return 1e-7f;
		}

		//  Evaluate tol
		template <typename T>
		__DEVICE__  T Tol();
		template <>
		__DEVICE__  double Tol()
		{
			return 1e-10;
		}
		template <>
		__DEVICE__  float Tol()
		{
			return 1e-5f;
		}

		// Evalution of max(0, x).
		template <typename T>
		__DEVICE__  T MaxPos(T x)
		{
			return Max(static_cast<T>(0), x);
		}

		//  Evalution of max(0, -x).
		template <typename T>
		__DEVICE__  T MaxNeg(T x)
		{
			return Max(static_cast<T>(0), -x);
		}

		//  Evalution of sign(x)
		template <typename T>
		__DEVICE__  T Sign(T x)
		{
			return x >= 0 ? 1 : -1;
		}

		// LambertW(Exp(x))
		// Evaluate the principal branch of the Lambert W function.
		// ref: http://keithbriggs.info/software/LambertW.c
		template <typename T>
		__DEVICE__  T LambertWExp(T x)
		{
			T w;
			if (x > static_cast<T>(100))
			{
				// Approximation for x in [100, 700].
				T log_x = Log(x);
				return static_cast<T>(-0.36962844)
					+ x
					- static_cast<T>(0.97284858) * log_x
					+ static_cast<T>(1.3437973) / log_x;
			}
			else if (x < static_cast<T>(0))
			{
				T p = Sqrt(static_cast<T>(2.0) * (Exp(x + static_cast<T>(1)) + static_cast<T>(1)));
				w = static_cast<T>(-1.0)
					+ p * (static_cast<T>(1.0)
					+ p * (static_cast<T>(-1.0 / 3.0)
					+ p * static_cast<T>(11.0 / 72.0)));
			}
			else
			{
				w = x;
			}
			if (x > static_cast<T>(1.098612288668110))
			{
				w -= Log(w);
			}
			for (unsigned int i = 0u; i < 10u; i++)
			{
				T e = Exp(w);
				T t = w * e - Exp(x);
				T p = w + static_cast<T>(1.);
				t /= e * p - static_cast<T>(0.5) * (p + static_cast<T>(1.0)) * t / p;
				w -= t;
				if (Abs(t) < Epsilon<T>() * (static_cast<T>(1) + Abs(w)))
					break;
			}
			return w;
		}

		// Find the root of a cubic x^3 + px^2 + qx + r = 0 with a single positive root.
		// ref: http://math.stackexchange.com/questions/60376
		template <typename T>
		__DEVICE__  T CubicSolve(T p, T q, T r)
		{
			T s = p / 3, s2 = s * s, s3 = s2 * s;
			T a = -s2 + q / 3;
			T b = s3 - s * q / 2 + r / 2;
			T a3 = a * a * a;
			T b2 = b * b;
			if (a3 + b2 >= 0)
			{
				T A = Pow(Sqrt(a3 + b2) - b, static_cast<T>(1) / 3);
				return -s - a / A + A;
			}
			else
			{
				T A = Sqrt(-a3);
				T B = Acos(-b / A);
				T C = Pow(A, static_cast<T>(1) / 3);
				return -s + (C - a / C) * Cos(B / 3);
			}
		}
		// Function definitions.
		//
		// Each of the following functions corresponds to one of the Function enums.
		// All functions accept one argument x and four parameters (a, b, c, and d)
		// and returns the evaluation of
		//
		//   x -> c * f(a * x - b) + d * x.
		template <typename T>
		__DEVICE__  T FuncAbs(T x)
		{
			return Abs(x);
		}

		template <typename T>
		__DEVICE__  T FuncNegEntr(T x)
		{
			return x <= 0 ? 0 : x * Log(x);
		}

		template <typename T>
		__DEVICE__  T FuncExp(T x)
		{
			return Exp(x);
		}

		template <typename T>
		__DEVICE__  T FuncHuber(T x)
		{
			T xabs = Abs(x);
			T xabs2 = xabs * xabs;
			return xabs < static_cast<T>(1) ? xabs2 / 2 : xabs - static_cast<T>(0.5);
		}

		template <typename T>
		__DEVICE__  T FuncIdentity(T x)
		{
			return x;
		}

		template <typename T>
		__DEVICE__  T FuncIndBox01(T x)
		{
			return 0;
		}

		template <typename T>
		__DEVICE__  T FuncIndEq0(T x)
		{
			return 0;
		}

		template <typename T>
		__DEVICE__  T FuncIndGe0(T x)
		{
			return 0;
		}

		template <typename T>
		__DEVICE__  T FuncIndLe0(T x)
		{
			return 0;
		}

		template <typename T>
		__DEVICE__  T FuncLogistic(T x)
		{
			return Log(1 + Exp(x));
		}

		template <typename T>
		__DEVICE__  T FuncMaxNeg0(T x)
		{
			return MaxNeg(x);
		}

		template <typename T>
		__DEVICE__  T FuncMaxPos0(T x)
		{
			return MaxPos(x);
		}

		template <typename T>
		__DEVICE__  T FuncNegLog(T x)
		{
			x = Max(static_cast<T>(0), x);
			return -Log(x);
		}

		template <typename T>
		__DEVICE__  T FuncRecpr(T x)
		{
			x = Max(static_cast<T>(0), x);
			return 1 / x;
		}

		template <typename T>
		__DEVICE__  T FuncSquare(T x)
		{
			return x * x / 2;
		}

		template <typename T>
		__DEVICE__  T FuncZero(T x)
		{
			return 0;
		}
		
	}

	template<typename T>
	__DEVICE__  T FuncEval(const FunctionObj<T> &f_obj, T x)
	{
		T dx = f_obj.d * x;
		T ex = f_obj.e * x * x / 2;
		x = f_obj.a * x - f_obj.b;
		switch (f_obj.h)
		{
		case UnaryFunc::Abs: x = functions::FuncAbs(x); break;
		case UnaryFunc::NegEntr: x = functions::FuncNegEntr(x); break;
		case UnaryFunc::Exp: x = functions::FuncExp(x); break;
		case UnaryFunc::Huber: x = functions::FuncHuber(x); break;
		case UnaryFunc::Identity: x = functions::FuncIdentity(x); break;
		case UnaryFunc::IndBox01: x = functions::FuncIndBox01(x); break;
		case UnaryFunc::IndEq0: x = functions::FuncIndEq0(x); break;
		case UnaryFunc::IndGe0: x = functions::FuncIndGe0(x); break;
		case UnaryFunc::IndLe0: x = functions::FuncIndLe0(x); break;
		case UnaryFunc::Logistic: x = functions::FuncLogistic(x); break;
		case UnaryFunc::MaxNeg0: x = functions::FuncMaxNeg0(x); break;
		case UnaryFunc::MaxPos0: x = functions::FuncMaxPos0(x); break;
		case UnaryFunc::NegLog: x = functions::FuncNegLog(x); break;
		case UnaryFunc::Recipr: x = functions::FuncRecpr(x); break;
		case UnaryFunc::Square: x = functions::FuncSquare(x); break;
		case UnaryFunc::Zero: default: x = functions::FuncZero(x); break;
		}
		return f_obj.c * x + dx + ex;
	}
} 
#endif
