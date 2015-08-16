#ifndef __H_enum_H__
#define __H_enum_H__
namespace alsm
{
	enum class DeviceType
	{
		CPU,
		GPU
	};
	enum class MatrixMemOrd
	{
		ROW=0,
		COL
	};
	enum class MatrixTrans
	{
		NORMAL=0,
		TRANSPOSE
	};
	enum class MatrixFillLower
	{
		UPPER=0,
		LOWER
	};
	enum class alsmStatus
	{
		alsm_SUCCESS,    // Converged succesfully.
		alsm_INFEASIBLE, // Problem likely infeasible.
		alsm_UNBOUNDED,  // Problem likely unbounded
		alsm_MAX_ITER,   // Reached max iter.
		alsm_NAN_FOUND,  // Encountered nan.
		alsm_ERROR
	};
	enum class UnaryFunc
	{
		Abs,       // f(x) = |x|
		Exp,       // f(x) = e^x
		Huber,     // f(x) = huber(x)
		Identity,  // f(x) = x
		IndBox01,  // f(x) = I(0 <= x <= 1)
		IndEq0,    // f(x) = I(x = 0)
		IndGe0,    // f(x) = I(x >= 0)
		IndLe0,    // f(x) = I(x <= 0)
		Logistic,  // f(x) = log(1 + e^x)
		MaxNeg0,   // f(x) = max(0, -x)
		MaxPos0,   // f(x) = max(0, x)
		NegEntr,   // f(x) = x log(x)
		NegLog,    // f(x) = -log(x)
		Recipr,    // f(x) = 1/x
		Square,    // f(x) = (1/2) x^2
		Zero
	};
	enum class DataFlowDirection
	{
		TOHOST,
		TODEVICE
	};
	enum class StopCriteria
	{
		ground_truth,//norm(x-xG)<=tol
		duality_gap,//norm(b-sum(Ax))/norm(b)<=tol
		objective_value,//(object_value(x)-object_value(x_prev))/object_value(x_prev)<=tol
		dual_tol,//duality_gap and beta*max(eta*norm(x-x_prev))/norm(b)<=tol_2
		increment,//norm(x_prev-x)<=tol*norm(x_prev)


	};
}
#endif
