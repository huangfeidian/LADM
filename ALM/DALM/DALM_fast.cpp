/*
% Copyright ©2010. The Regents of the University of California (Regents).
% All Rights Reserved. Contact The Office of Technology Licensing,
% UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
% (510) 643-7201, for commercial licensing opportunities.

% Authors: Victor Shia, Mark Murphy and Allen Y. Yang.
% Contact: Allen Y. Yang, Department of EECS, University of California,
% Berkeley. <yang@eecs.berkeley.edu>

% IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
% SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
% ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
% REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED
% TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
% PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY,
% PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO
% PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include <string.h>
#include <math.h>
#include <float.h>
#include <cublas.h>
#include <stdio.h>
#include <malloc.h>
#include <algorithm>
float const eps = FLT_EPSILON;
#include "DALM_fast.h"
#if !defined (COMPILE_MEX)

#undef mexPrintf
#define mexPrintf printf

#else
#include <mex.h>

void SolveDALM_fast(
	float *&x, int &nIter,
	const float *b, const float *A, float lambda, float tol, int m, int n, int stoppingCriterion, const float *xG);

extern "C" void
mexFunction(int nl, mxArray *pl [], int nr, mxArray const *pr [])
{
	if (nr < 4)
	{
		mexErrMsgTxt("[x nIter] = SolveDALM_fast(A, b, nu, tol, stop, xG)");
	}

	float *A = (float *) mxGetData(pr[0]);
	int m = mxGetM(pr[0]);
	int n = mxGetN(pr[0]);
	float *b = (float *) mxGetData(pr[1]);

	if (mxGetM(pr[1]) * mxGetN(pr[1]) != m)
	{
		mexErrMsgTxt("SolveDALM_fast: min |x|1 + |e|1 s.t. Ax + e = b\n");
	}

	float nu = (float) mxGetScalar(pr[2]);
	float tol = (float) mxGetScalar(pr[3]);
	float *xG;	int stop;

	if (nr < 6)
		xG = NULL;
	else
		xG = (float *) mxGetData(pr[5]);
	if (nr < 5)
		stop = 5;
	else
		stop = (int) mxGetScalar(pr[4]);

	/*
	mexPrintf("Arguments: %d\n", nr);
	mexPrintf("Stopping criterion: %d\n", stop);
	mexPrintf("ground_x:\n");
	for(int k = 0 ; k < n; k++){
	mexPrintf("%f ", xG[k]);
	}
	mexPrintf("\n");
	mexPrintf("b:");
	for(int k = 0 ; k < m; k++){
	mexPrintf("%f ", b[k]);
	}
	mexPrintf("\n");
	*/

	float *x;
	int nIter;

	SolveDALM_fast(x, nIter, b, A, nu, tol, m, n, stop, xG);
	//	norm_x_e_dual (x, e, nIter, b, A, nu, tol, m, n);

	if (nl > 0)
	{
		pl[0] = mxCreateNumericMatrix(n, 1, mxSINGLE_CLASS, mxREAL);
		memcpy(mxGetData(pl[0]), (void*) x, n*sizeof(float));
	}
	delete [] x;

	if (nl > 1)
	{
		pl[1] = mxCreateDoubleScalar(nIter);
	}
}

#endif


// z = sign(temp1).*min(1,abs(temp1));
extern void cudaS_sign_min(int grid_size, int block_size,const  float* temp1, float* z, int len);
// beta * (temp - z) + x

extern void cudaS_btzx(int grid_size, int block_size, float beta, const float* temp, const float* z, const float* x, float* rtn, int len);

//xbzt<<< grid_size, block_size >>>(d_x, beta, z, temp, tmp, n); // MY OWN FUNCTION
//	    x = x - beta * (z - temp);

extern void cudaS_xbzt(int grid_size, int block_size, const  float* x, float b, const float* z, const float* temp, float* rtn, int len);

enum stoppingCriteria
{
	STOPPING_GROUND_TRUTH = -1,
	STOPPING_DUALITY_GAP = 1,
	STOPPING_SPARSE_SUPPORT = 2,
	STOPPING_OBJECTIVE_VALUE = 3,
	STOPPING_SUBGRADIENT = 4,
	STOPPING_INCREMENTS = 5,
	STOPPING_GROUND_OBJECT=6,
	STOPPING_KKT_DUAL_TOL = 7,
	STOPPING_LOG = 8,//no stop just to log statistic informations
	STOPPING_DEFAULT = STOPPING_INCREMENTS
};
dalmSsolver::dalmSsolver(int in_m, int in_n, int in_stop, float in_tol, float in_lambda,float in_tol2,int in_max_iter)
	:m(in_m), n(in_n), stoppingCriterion(in_stop), tol(in_tol), lambda(in_lambda), tol2(in_tol2), maxIter(in_max_iter)
{
	
	ldA = m;
	nIter = 0;
	verbose = false;
	result = 0;
	f = 1;
	switch (stoppingCriterion)
	{
	case -1:
		stop = STOPPING_GROUND_TRUTH;
		break;
	case 1:
		stop = STOPPING_DUALITY_GAP;
		break;
	case 2:
		stop = STOPPING_SPARSE_SUPPORT;
		break;
	case 3:
		stop = STOPPING_OBJECTIVE_VALUE;
		break;
	case 4:
		stop = STOPPING_SUBGRADIENT;
		break;
	case 5:
		stop = STOPPING_INCREMENTS;
		break;
	case 6:
		stop = STOPPING_GROUND_OBJECT;
		break;
	case 7:
		stop = STOPPING_KKT_DUAL_TOL;
		break;
	case 8:
		stop = STOPPING_LOG;
		break;

	}

	
	

	
}
void dalmSsolver::allocate_memory()
{
	cudaDeviceProp properties;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1)
	{
		int max_multiprocessors = 0, max_device = 0;
		for (device = 0; device < num_devices; device++)
		{
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount)
			{
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		max_device = num_devices - 1;
		cudaSetDevice(max_device);
		cudaGetDeviceProperties(&properties, max_device);
		////mexPrintf("GPU Processor %d", max_device);
	}
	else
	{
		cudaGetDeviceProperties(&properties, 0);
		////mexPrintf("GPU Processor %d", 0);
	}
	max_threads = properties.maxThreadsPerBlock;
	cublasStatus stat;

	stat = cublasInit();


	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		mexPrintf("ERROR: %d", stat);
		return;
	}

	cublasAlloc(m, sizeof(float), (void**) &d_b);

	cublasAlloc(m*n, sizeof(float), (void**) &d_A);

	cublasAlloc(m, sizeof(float), (void**) &y);
	cublasAlloc(m, sizeof(float), (void**) &diff_b);
	cublasAlloc(n, sizeof(float), (void**) &d_x);
	cublasAlloc(n, sizeof(float), (void**) &z);
	cublasAlloc(std::max(m, n), sizeof(float), (void**) &temp);
	cublasAlloc(n, sizeof(float), (void**) &Ag);
	cublasAlloc(n, sizeof(float), (void**) &x_old);
	cublasAlloc(std::max(m, n), sizeof(float), (void**) &temp1);
	cublasAlloc(std::max(m, n), sizeof(float), (void**) &tmp);
	cublasAlloc(m, sizeof(float), (void**) &g);
	cublasAlloc(n, sizeof(float), (void**) &d_xG);
}
void dalmSsolver::set_log_file(FILE* input_file)
{
	log_file = input_file;
	// nIter, norm(diff(x)),norm(diff(x))/norm(prev_x),residual,residual/b,f,eps(diff(f))
	fprintf(log_file, "nIter,norm(diff(x)),eps(diff(x)),residual,eps(residual),f,eps(diff(f))\n");
}
void   dalmSsolver::solve(float *x,  const float *b, const float *A, const float *xG)
{



	if (stop == STOPPING_GROUND_TRUTH||stop==STOPPING_GROUND_OBJECT)
	{
		cublasSetVector(n, sizeof(float), xG, 1, d_xG, 1);
	}

	cublasSetVector(m, sizeof(float), b, 1, d_b, 1);
	cublasSetMatrix(m, n, sizeof(float), A, m, d_A, m);
	float nrm_b,diff_nrm_b;
	//	beta = norm(b,1)/m;
	//	betaInv = 1/beta ;
	beta = cublasSasum(m, d_b, 1) / m;
	nrm_b = cublasSnrm2(m,d_b,1);
	betaInv = 1 / beta;

	//	nIter = 0 ;
	nIter = 0;

	//	y = zeros(m,1);
	//	x = zeros(n,1);    
	//	z = zeros(m+n,1);
	cudaMemset(y, 0, m*sizeof(float));
	cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
	cublasSetVector(n, sizeof(float), x, 1, z, 1);

	//	converged_main = 0 ;
	converged_main = false;

	//	temp = A' * y;
	cublasSgemv('T', m, n, 1.0, d_A, ldA, y, 1, 0.0, temp, 1);
	if (verbose)
	{
		mexPrintf("sasum(A): %20.20f\n", cublasSasum(m*n, d_A, 1));
		mexPrintf("sasum(b): %20.20f\n", cublasSasum(m, d_b, 1));
		mexPrintf("sasum(temp): %20.20f\n", cublasSasum(n, temp, 1));
	}

	//	f = norm(x,1);
	f = 1;
	prev_f = 0;
	total = 0;
	if (stop == STOPPING_GROUND_OBJECT)
	{
		prev_f = cublasSasum( n, d_xG, 1);
	}
	int block_size = max_threads;
	int grid_size = (int) (std::max(n, m) / max_threads) + 1;

	//	int block_size = 512;
	//	int grid_size  = (max(m,n) + block_size - 1) / block_size;

	do
	{
		//      nIter = nIter + 1 ;  
		nIter++;
		if (verbose) mexPrintf("==== [%d] ====\n", nIter);

		//		x_old = x;
		cublasScopy(n, d_x, 1, x_old, 1);

		//	    %update z
		//	    temp1 = temp + x * betaInv;
		//	    z = sign(temp1) .* min(1,abs(temp1));
		cublasScopy(n, temp, 1, temp1, 1);
		cublasSaxpy(n, betaInv, d_x, 1, temp1, 1);
		cudaS_sign_min(grid_size, block_size ,temp1, z, n);  // MY OWN FUNCTION

		//		%compute A' * y    
		//	    g = lambda * y - b + A * (beta * (temp - z) + x);
		cudaS_btzx(grid_size, block_size,beta, temp, z, d_x, tmp, n);  // MY OWN FUNCTION
		if (verbose)
		{
			mexPrintf("beta: %20.20f\n", beta);
			mexPrintf("sasum(d_x): %20.20f\n", cublasSasum(n, d_x, 1));
			mexPrintf("sasum(z): %20.20f\n", cublasSasum(n, z, 1));
			mexPrintf("sasum(temp): %20.20f\n", cublasSasum(n, temp, 1));
			mexPrintf("sasum(tmp): %20.20f\n", cublasSasum(n, tmp, 1));
		}
		cublasScopy(m, d_b, 1, g, 1);
		cublasSaxpy(m, -1 * lambda, y, 1, g, 1);
		cublasSgemv('N', m, n, 1.0, d_A, ldA, tmp, 1, -1, g, 1);

		//		%alpha = g' * g / (g' * G * g);
		//	    Ag = A' * g;
		cublasSgemv('T', m, n, 1.0, d_A, ldA, g, 1, 0.0, Ag, 1);

		//	    alpha = g' * g / (lambda * g' * g + beta * Ag' * Ag);
		dg = cublasSdot(m, g, 1, g, 1);
		dAg = cublasSdot(n, Ag, 1, Ag, 1);
		alpha = dg / (lambda * dg + beta * dAg);

		//	    y = y - alpha * g;
		cublasSaxpy(m, -1 * alpha, g, 1, y, 1);

		//	    temp = A' * y;
		cublasSgemv('T', m, n, 1.0, d_A, ldA, y, 1, 0.0, temp, 1);

		//	    %update x
		//	    x = x - beta * (z - temp);
		//		xbzt<<< grid_size, block_size >>>(d_x, beta, z, temp, tmp, n); // MY OWN FUNCTION
		cudaS_btzx(grid_size, block_size ,-1 * beta, z, temp, d_x, tmp, n);
		cublasScopy(n, tmp, 1, d_x, 1);

		if (verbose)
		{
			mexPrintf("sasum(z): %20.20f\n", cublasSasum(n, z, 1));
			mexPrintf("sasum(g): %20.20f\n", cublasSasum(n, g, 1));
			mexPrintf("sasum(y): %20.20f\n", cublasSasum(n, y, 1));
			mexPrintf("sasum(temp): %20.20f\n", cublasSasum(n, temp, 1));
			mexPrintf("sasum(x): %20.20f\n", cublasSasum(n, d_x, 1));
			mexPrintf("beta: %20.20f\n", beta);
		}
		// STOPPING CRITERION
		switch (stop)
		{
		case STOPPING_GROUND_TRUTH:
			//        if norm(xG-x) < tol
			//            converged_main = 1 ;
			//        end
			cublasScopy(n, d_xG, 1, tmp, 1);
			cublasSaxpy(n, -1, d_x, 1, tmp, 1);
			total = cublasSnrm2(n, tmp, 1);

			mexPrintf("total: %f", total);
			mexPrintf("tol: %f", tol);

			if (total < tol)
				converged_main = true;
			break;
		case STOPPING_SUBGRADIENT:
			mexPrintf("Duality gap is not a valid stopping criterion for ALM.");
			break;
		case STOPPING_SPARSE_SUPPORT:
			mexPrintf("DALM does not have a support set.");
			break;
		case STOPPING_OBJECTIVE_VALUE:
			//          prev_f = f;
			//          f = norm(x,1);
			//          criterionObjective = abs(f-prev_f)/(prev_f);
			//          converged_main =  ~(criterionObjective > tol);
			prev_f = f;
			f = cublasSasum(n, d_x, 1);
			if (fabs(f - prev_f)  <= tol*prev_f)
			{
				converged_main = true;
			}
			break;
		case STOPPING_DUALITY_GAP:
			cublasSgemv('N', m, n, 1, d_A, m, d_x, 1, 0, diff_b, 1);
			cublasSaxpy(m, -1, d_b, 1, diff_b, 1);
			diff_nrm_b = cublasSasum(m, diff_b, 1);
			if (diff_nrm_b  < tol*nrm_b)
			{
				converged_main = true;
			}
			//mexPrintf("Duality gap is not a valid stopping criterion for ALM.");
			break;
		case STOPPING_INCREMENTS:
			// if norm(x_old - x) < tol * norm(x_old)
			//     converged_main = true;
			nxo = cublasSnrm2(n, x_old, 1);
			cublasSaxpy(n, -1, d_x, 1, x_old, 1);
			dx = cublasSnrm2(n, x_old, 1);

			if (dx < tol*nxo)
				converged_main = true;

			if (verbose)
			{
				if (nIter > 1)
				{
					mexPrintf("  ||dx|| = %f (= %f * ||x_old||)\n",
						dx, dx / (nxo + eps));
				}
				else
				{
					mexPrintf("  ||dx|| = %f\n", dx);
					mexPrintf("  ||tol|| = %f\n", tol);
					mexPrintf("  ||nxo|| = %f\n", nxo);
				}
			}
			break;
		case STOPPING_GROUND_OBJECT:
			f = cublasSasum(n, d_x, 1);
			if (fabs(f - prev_f)<= tol*prev_f)
			{
				converged_main = true;
			}
			break;
		case STOPPING_KKT_DUAL_TOL:
			nxo = cublasSnrm2(n, x_old, 1);
			cublasSaxpy(n, -1, d_x, 1, x_old, 1);
			dx = cublasSnrm2(n, x_old, 1);
			cublasSgemv('N', m, n, 1, d_A, m, d_x, 1, 0, diff_b, 1);
			cublasSaxpy(m, -1, d_b, 1, diff_b, 1);
			diff_nrm_b = cublasSasum(m, diff_b, 1);
			if (diff_nrm_b  < tol*nrm_b&&dx<tol2*nxo)
			{
				converged_main = true;
			}
			break;
		case STOPPING_LOG:
			prev_f = f;
			f = cublasSasum(n, d_x, 1);
			nxo = cublasSnrm2(n, x_old, 1);
			cublasSaxpy(n, -1, d_x, 1, x_old, 1);
			dx = cublasSnrm2(n, x_old, 1);
			cublasSgemv('N', m, n, 1, d_A, m, d_x, 1, 0, diff_b, 1);
			cublasSaxpy(m, -1, d_b, 1, diff_b, 1);
			diff_nrm_b = cublasSasum(m, diff_b, 1);
			fprintf(log_file, "%lf,%lf,%lf,", static_cast<double>(nIter), static_cast<double>(dx), static_cast<double>(dx / nxo));
			fprintf(log_file, "%lf,%lf,%lf,%lf\n", static_cast<double>(diff_nrm_b), static_cast<double>(diff_nrm_b / nrm_b), static_cast<double>(f), static_cast<double>(fabs((prev_f - f) / prev_f)));

			break;
		default:
			mexPrintf("Undefined stopping criterion.");
			break;
		}

		if (nIter >= maxIter)
		{
			if (verbose) mexPrintf("Max Iterations Reached\n");
			converged_main = true;
		}

	}
	while (!converged_main);

	mexPrintf("==== CONVERGED in iteration %d ==== \n", nIter);

	result = cublasSasum(n, d_x, 1);
	cublasGetVector(n, sizeof(float), d_x, 1, x, 1);
	
	
}
void dalmSsolver::free_memory()
{
	cublasFree(d_A);
	cublasFree(d_b);
	cublasFree(d_x);
	cublasFree(tmp);
	cublasFree(g);
	cublasFree(Ag);
	cublasFree(y);
	cublasFree(z);
	cublasFree(x_old);
	cublasFree(temp);
	cublasFree(temp1);
	cublasFree(d_xG);


	cublasShutdown();
}
// z = sign(temp1).*min(1,abs(temp1));
extern void cudaD_sign_min(int grid_size, int block_size, const  double* temp1, double* z, int len);
// beta * (temp - z) + x

extern void cudaD_btzx(int grid_size, int block_size, double beta, const double* temp, const double* z, const double* x, double* rtn, int len);

//xbzt<<< grid_size, block_size >>>(d_x, beta, z, temp, tmp, n); // MY OWN FUNCTION
//	    x = x - beta * (z - temp);

extern void cudaD_xbzt(int grid_size, int block_size, const  double* x, double b, const double* z, const double* temp, double* rtn, int len);
dalmDsolver::dalmDsolver(int in_m, int in_n, int in_stop, double in_tol, double in_lambda, double in_tol2, int in_max_iter)
	:m(in_m), n(in_n), stoppingCriterion(in_stop), tol(in_tol), lambda(in_lambda), tol2(in_tol2), maxIter(in_max_iter)
{

	ldA = m;
	nIter = 0;
	verbose = false;
	result = 0;
	f = 1;
	switch (stoppingCriterion)
	{
	case -1:
		stop = STOPPING_GROUND_TRUTH;
		break;
	case 1:
		stop = STOPPING_DUALITY_GAP;
		break;
	case 2:
		stop = STOPPING_SPARSE_SUPPORT;
		break;
	case 3:
		stop = STOPPING_OBJECTIVE_VALUE;
		break;
	case 4:
		stop = STOPPING_SUBGRADIENT;
		break;
	case 5:
		stop = STOPPING_INCREMENTS;
		break;
	case 6:
		stop = STOPPING_GROUND_OBJECT;
		break;
	case 7:
		stop = STOPPING_KKT_DUAL_TOL;
		break;
	case 8:
		stop = STOPPING_LOG;
		break;

	}





}
void dalmDsolver::allocate_memory()
{
	cudaDeviceProp properties;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1)
	{
		int max_multiprocessors = 0, max_device = 0;
		for (device = 0; device < num_devices; device++)
		{
			cudaGetDeviceProperties(&properties, device);
			if (max_multiprocessors < properties.multiProcessorCount)
			{
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
		max_device = num_devices - 1;
		cudaSetDevice(max_device);
		cudaGetDeviceProperties(&properties, max_device);
		////mexPrintf("GPU Processor %d", max_device);
	}
	else
	{
		cudaGetDeviceProperties(&properties, 0);
		////mexPrintf("GPU Processor %d", 0);
	}
	max_threads = properties.maxThreadsPerBlock;
	cublasStatus stat;

	stat = cublasInit();


	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		mexPrintf("ERROR: %d", stat);
		return;
	}

	cublasAlloc(m, sizeof(double), (void**) &d_b);

	cublasAlloc(m*n, sizeof(double), (void**) &d_A);

	cublasAlloc(m, sizeof(double), (void**) &y);
	cublasAlloc(m, sizeof(double), (void**) &diff_b);
	cublasAlloc(n, sizeof(double), (void**) &d_x);
	cublasAlloc(n, sizeof(double), (void**) &z);
	cublasAlloc(std::max(m, n), sizeof(double), (void**) &temp);
	cublasAlloc(n, sizeof(double), (void**) &Ag);
	cublasAlloc(n, sizeof(double), (void**) &x_old);
	cublasAlloc(std::max(m, n), sizeof(double), (void**) &temp1);
	cublasAlloc(std::max(m, n), sizeof(double), (void**) &tmp);
	cublasAlloc(m, sizeof(double), (void**) &g);
	cublasAlloc(n, sizeof(double), (void**) &d_xG);
}
void dalmDsolver::set_log_file(FILE* input_file)
{
	log_file = input_file;
	// nIter, norm(diff(x)),norm(diff(x))/norm(prev_x),residual,residual/b,f,eps(diff(f))
	fprintf(log_file, "nIter,norm(diff(x)),eps(diff(x)),residual,eps(residual),f,eps(diff(f))\n");
}
void   dalmDsolver::solve(double *x, const double *b, const double *A, const double *xG)
{



	if (stop == STOPPING_GROUND_TRUTH || stop == STOPPING_GROUND_OBJECT)
	{
		cublasSetVector(n, sizeof(double), xG, 1, d_xG, 1);
	}

	cublasSetVector(m, sizeof(double), b, 1, d_b, 1);
	cublasSetMatrix(m, n, sizeof(double), A, m, d_A, m);
	double nrm_b, diff_nrm_b;
	//	beta = norm(b,1)/m;
	//	betaInv = 1/beta ;
	beta = cublasDasum(m, d_b, 1) / m;
	nrm_b = cublasDnrm2(m, d_b, 1);
	betaInv = 1 / beta;

	//	nIter = 0 ;
	nIter = 0;

	//	y = zeros(m,1);
	//	x = zeros(n,1);    
	//	z = zeros(m+n,1);
	cudaMemset(y, 0, m*sizeof(double));
	cublasSetVector(n, sizeof(double), x, 1, d_x, 1);
	cublasSetVector(n, sizeof(double), x, 1, z, 1);

	//	converged_main = 0 ;
	converged_main = false;

	//	temp = A' * y;
	cublasDgemv('T', m, n, 1.0, d_A, ldA, y, 1, 0.0, temp, 1);
	if (verbose)
	{
		mexPrintf("sasum(A): %20.20f\n", cublasDasum(m*n, d_A, 1));
		mexPrintf("sasum(b): %20.20f\n", cublasDasum(m, d_b, 1));
		mexPrintf("sasum(temp): %20.20f\n", cublasDasum(n, temp, 1));
	}

	//	f = norm(x,1);
	f = 1;
	prev_f = 0;
	total = 0;
	if (stop == STOPPING_GROUND_OBJECT)
	{
		prev_f = cublasDasum(n, d_xG, 1);
	}
	int block_size = max_threads;
	int grid_size = (int) (std::max(n, m) / max_threads) + 1;

	//	int block_size = 512;
	//	int grid_size  = (max(m,n) + block_size - 1) / block_size;

	do
	{
		//      nIter = nIter + 1 ;  
		nIter++;
		if (verbose) mexPrintf("==== [%d] ====\n", nIter);

		//		x_old = x;
		cublasDcopy(n, d_x, 1, x_old, 1);

		//	    %update z
		//	    temp1 = temp + x * betaInv;
		//	    z = sign(temp1) .* min(1,abs(temp1));
		cublasDcopy(n, temp, 1, temp1, 1);
		cublasDaxpy(n, betaInv, d_x, 1, temp1, 1);
		cudaD_sign_min(grid_size, block_size, temp1, z, n);  // MY OWN FUNCTION

		//		%compute A' * y    
		//	    g = lambda * y - b + A * (beta * (temp - z) + x);
		cudaD_btzx(grid_size, block_size, beta, temp, z, d_x, tmp, n);  // MY OWN FUNCTION
		if (verbose)
		{
			mexPrintf("beta: %20.20f\n", beta);
			mexPrintf("sasum(d_x): %20.20f\n", cublasDasum(n, d_x, 1));
			mexPrintf("sasum(z): %20.20f\n", cublasDasum(n, z, 1));
			mexPrintf("sasum(temp): %20.20f\n", cublasDasum(n, temp, 1));
			mexPrintf("sasum(tmp): %20.20f\n", cublasDasum(n, tmp, 1));
		}
		cublasDcopy(m, d_b, 1, g, 1);
		cublasDaxpy(m, -1 * lambda, y, 1, g, 1);
		cublasDgemv('N', m, n, 1.0, d_A, ldA, tmp, 1, -1, g, 1);

		//		%alpha = g' * g / (g' * G * g);
		//	    Ag = A' * g;
		cublasDgemv('T', m, n, 1.0, d_A, ldA, g, 1, 0.0, Ag, 1);

		//	    alpha = g' * g / (lambda * g' * g + beta * Ag' * Ag);
		dg = cublasDdot(m, g, 1, g, 1);
		dAg = cublasDdot(n, Ag, 1, Ag, 1);
		alpha = dg / (lambda * dg + beta * dAg);

		//	    y = y - alpha * g;
		cublasDaxpy(m, -1 * alpha, g, 1, y, 1);

		//	    temp = A' * y;
		cublasDgemv('T', m, n, 1.0, d_A, ldA, y, 1, 0.0, temp, 1);

		//	    %update x
		//	    x = x - beta * (z - temp);
		//		xbzt<<< grid_size, block_size >>>(d_x, beta, z, temp, tmp, n); // MY OWN FUNCTION
		cudaD_btzx(grid_size, block_size, -1 * beta, z, temp, d_x, tmp, n);
		cublasDcopy(n, tmp, 1, d_x, 1);

		if (verbose)
		{
			mexPrintf("sasum(z): %20.20f\n", cublasDasum(n, z, 1));
			mexPrintf("sasum(g): %20.20f\n", cublasDasum(n, g, 1));
			mexPrintf("sasum(y): %20.20f\n", cublasDasum(n, y, 1));
			mexPrintf("sasum(temp): %20.20f\n", cublasDasum(n, temp, 1));
			mexPrintf("sasum(x): %20.20f\n", cublasDasum(n, d_x, 1));
			mexPrintf("beta: %20.20f\n", beta);
		}
		// STOPPING CRITERION
		switch (stop)
		{
		case STOPPING_GROUND_TRUTH:
			//        if norm(xG-x) < tol
			//            converged_main = 1 ;
			//        end
			cublasDcopy(n, d_xG, 1, tmp, 1);
			cublasDaxpy(n, -1, d_x, 1, tmp, 1);
			total = cublasDnrm2(n, tmp, 1);

			mexPrintf("total: %f", total);
			mexPrintf("tol: %f", tol);

			if (total < tol)
				converged_main = true;
			break;
		case STOPPING_SUBGRADIENT:
			mexPrintf("Duality gap is not a valid stopping criterion for ALM.");
			break;
		case STOPPING_SPARSE_SUPPORT:
			mexPrintf("DALM does not have a support set.");
			break;
		case STOPPING_OBJECTIVE_VALUE:
			//          prev_f = f;
			//          f = norm(x,1);
			//          criterionObjective = abs(f-prev_f)/(prev_f);
			//          converged_main =  ~(criterionObjective > tol);
			prev_f = f;
			f = cublasDasum(n, d_x, 1);
			if (fabs(f - prev_f) <= tol*prev_f)
			{
				converged_main = true;
			}
			break;
		case STOPPING_DUALITY_GAP:
			cublasDgemv('N', m, n, 1, d_A, m, d_x, 1, 0, diff_b, 1);
			cublasDaxpy(m, -1, d_b, 1, diff_b, 1);
			diff_nrm_b = cublasDasum(m, diff_b, 1);
			if (diff_nrm_b  < tol*nrm_b)
			{
				converged_main = true;
			}
			//mexPrintf("Duality gap is not a valid stopping criterion for ALM.");
			break;
		case STOPPING_INCREMENTS:
			// if norm(x_old - x) < tol * norm(x_old)
			//     converged_main = true;
			nxo = cublasDnrm2(n, x_old, 1);
			cublasDaxpy(n, -1, d_x, 1, x_old, 1);
			dx = cublasDnrm2(n, x_old, 1);

			if (dx < tol*nxo)
				converged_main = true;

			if (verbose)
			{
				if (nIter > 1)
				{
					mexPrintf("  ||dx|| = %f (= %f * ||x_old||)\n",
						dx, dx / (nxo + eps));
				}
				else
				{
					mexPrintf("  ||dx|| = %f\n", dx);
					mexPrintf("  ||tol|| = %f\n", tol);
					mexPrintf("  ||nxo|| = %f\n", nxo);
				}
			}
			break;
		case STOPPING_GROUND_OBJECT:
			f = cublasDasum(n, d_x, 1);
			if (fabs(f - prev_f) <= tol*prev_f)
			{
				converged_main = true;
			}
			break;
		case STOPPING_KKT_DUAL_TOL:
			nxo = cublasDnrm2(n, x_old, 1);
			cublasDaxpy(n, -1, d_x, 1, x_old, 1);
			dx = cublasDnrm2(n, x_old, 1);
			cublasDgemv('N', m, n, 1, d_A, m, d_x, 1, 0, diff_b, 1);
			cublasDaxpy(m, -1, d_b, 1, diff_b, 1);
			diff_nrm_b = cublasDasum(m, diff_b, 1);
			if (diff_nrm_b  < tol*nrm_b&&dx<tol2*nxo)
			{
				converged_main = true;
			}
			break;
		case STOPPING_LOG:
			prev_f = f;
			nxo = cublasDnrm2(n, x_old, 1);
			f = cublasDasum(n, d_x, 1);

			cublasDaxpy(n, -1, d_x, 1, x_old, 1);
			dx = cublasDnrm2(n, x_old, 1);
			cublasDgemv('N', m, n, 1, d_A, m, d_x, 1, 0, diff_b, 1);
			cublasDaxpy(m, -1, d_b, 1, diff_b, 1);
			diff_nrm_b = cublasDasum(m, diff_b, 1);
			fprintf(log_file, "%lf,%lf,%lf,", static_cast<double>(nIter), static_cast<double>(dx), static_cast<double>(dx / nxo));
			fprintf(log_file, "%lf,%lf,%lf,%lf\n", static_cast<double>(diff_nrm_b), static_cast<double>(diff_nrm_b / nrm_b), static_cast<double>(f), static_cast<double>(fabs((prev_f - f) / prev_f)));

			break;
		default:
			mexPrintf("Undefined stopping criterion.");
			break;
		}

		if (nIter >= maxIter)
		{
			if (verbose) mexPrintf("Max Iterations Reached\n");
			converged_main = true;
		}

	}
	while (!converged_main);

	mexPrintf("==== CONVERGED in iteration %d ==== \n", nIter);

	result = cublasDasum(n, d_x, 1);
	cublasGetVector(n, sizeof(double), d_x, 1, x, 1);


}
void dalmDsolver::free_memory()
{
	cublasFree(d_A);
	cublasFree(d_b);
	cublasFree(d_x);
	cublasFree(tmp);
	cublasFree(g);
	cublasFree(Ag);
	cublasFree(y);
	cublasFree(z);
	cublasFree(x_old);
	cublasFree(temp);
	cublasFree(temp1);
	cublasFree(d_xG);


	cublasShutdown();
}