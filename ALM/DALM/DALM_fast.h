#include <iostream>
class DALM_solver
{
public:
	int m;
	int n;
	int stoppingCriterion;
	float tol;
	float lambda;
	float result;
	int nIter;
	float prev_f;
	float f;
	float *x, *y, *d_x, *temp, *Ag, *x_old, *temp1, *tmp, *g, *d_b, *d_A, *d_xG;
	float *z;
	bool converged_main;
	int ldA;
	int maxIter ;
	float beta, betaInv, nxo, dx, total, dg, dAg, alpha;
	bool verbose ;

	enum stoppingCriteria stop;

	int num_devices, device, max_threads;
	DALM_solver(int in_m, int in_n, int in_stop, float in_tol, float in_lambda);
	void solve(float* &x,  const float* b, const float* A, const float* xG);
	void free_memory();
};