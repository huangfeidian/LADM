#include <iostream>
class DALM_solver
{
public:
	int m;
	int n;
	int stoppingCriterion;
	float tol;
	float tol2;
	float lambda;
	float result;
	int nIter;
	float prev_f;
	float f;
	float  *y, *d_x, *temp, *Ag, *x_old, *temp1, *tmp, *g, *d_b, *d_A, *d_xG;
	float *z;
	bool converged_main;
	int ldA;
	int maxIter ;
	float beta, betaInv, nxo, dx, total, dg, dAg, alpha;
	bool verbose ;
	enum stoppingCriteria stop;
	float* diff_b;
	int num_devices, device, max_threads;
	FILE* log_file;
	
	DALM_solver(int in_m, int in_n, int in_stop, float in_tol, float in_lambda,float in_tol2=0.001);
	void set_log_file(FILE* input_file);
	void allocate_memory();
	void solve(float* x,  const float* b, const float* A, const float* xG);
	void free_memory();
};