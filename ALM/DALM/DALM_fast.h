#include <iostream>
class dalmSsolver
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
	
	dalmSsolver(int in_m, int in_n, int in_stop, float in_tol, float in_lambda,float in_tol2=0.001,int in_max_iter=5000);
	void set_log_file(FILE* input_file);
	void allocate_memory();
	void solve(float* x,  const float* b, const float* A, const float* xG);
	void free_memory();
};
class dalmDsolver
{
public:
	int m;
	int n;
	int stoppingCriterion;
	double tol;
	double tol2;
	double lambda;
	double result;
	int nIter;
	double prev_f;
	double f;
	double  *y, *d_x, *temp, *Ag, *x_old, *temp1, *tmp, *g, *d_b, *d_A, *d_xG;
	double *z;
	bool converged_main;
	int ldA;
	int maxIter;
	double beta, betaInv, nxo, dx, total, dg, dAg, alpha;
	bool verbose;
	enum stoppingCriteria stop;
	double* diff_b;
	int num_devices, device, max_threads;
	FILE* log_file;

	dalmDsolver(int in_m, int in_n, int in_stop, double in_tol, double in_lambda, double in_tol2 = 0.001, int in_max_iter = 5000);
	void set_log_file(FILE* input_file);
	void allocate_memory();
	void solve(double* x, const double* b, const double* A, const double* xG);
	void free_memory();
};