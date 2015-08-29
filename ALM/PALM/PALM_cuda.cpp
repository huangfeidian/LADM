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
//#include "portable_blas_wrapper.h"
#include <stdio.h>
#include "PALM_cuda.h"
#include <cublas.h>
#include <algorithm>
#ifdef BLAS_IMPLEMENTATION_MKL
#undef int64_t
#define int64_t int
#endif


extern void cuda_scale_array(int grid_size, int block_size, const float *arr, int len, float a, float *rtn);
extern void cuda_calculate_lambda(int grid_size, int block_size, const float *lambda, float mu, const float *b, const float *temp, const float *e, float *rtn, int len);
extern void cuda_add_arrays(int gird_size, int block_size, const float *a, const float *b, float *c, int len);
extern void cuda_shrink(int grid_size, int block_size, const float *x, float alpha, float *y, int len);
extern void cuda_scale_sub_array(int grid_size, int block_size, const float *z, float tauInv, const float *temp1, const float *Gz, float *rtn, int len);
extern void cuda_manyOps(int grid_size, int block_size, float tau, const float *z, const float *x, const float *Gx, const float *Gz, float *rtn, int len);
extern void cuda_calculate_inner_val(int grid_size, int block_size, const float *x, float a, const  float *x_old_apg, float *z, int len);
extern void cuda_set_array(int grid_size, int block_size, const float *b, int len, float c);
extern void cuda_set_nzx(int grid_size, int block_size, bool *nz_x, const float *x, int len, float threshold);

enum stoppingCriteria
{
	STOPPING_GROUND_TRUTH = -1,
	STOPPING_DUALITY_GAP = 1,
	STOPPING_SPARSE_SUPPORT = 2,
	STOPPING_OBJECTIVE_VALUE = 3,
	STOPPING_SUBGRADIENT = 4,
	STOPPING_INCREMENTS = 5,
	STOPPING_GROUND_OBJECT = 6,
	STOPPING_KKT_DUAL_TOL=7,
	STOPPING_LOG=8,//no stop just to log statistic informations
	STOPPING_DEFAULT = STOPPING_INCREMENTS
};
L1Solver_e1x1::L1Solver_e1x1(int new_m, int new_n)
{
	m = new_m;
	n = new_n;
	prev_f = f = -1;
	// Initialize CUDA
	int num_devices, device, max_threads;
	cudaDeviceProp properties;
	cudaGetDeviceCount(&num_devices);
	//printf("num_devices: %d\n", num_devices);
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
		//////printf("GPU Processor %d", max_device);
		cudaSetDevice(0); // for mescal607
	}
	else
	{
		cudaGetDeviceProperties(&properties, 0);
		//////printf("GPU Processor %d", 0);
	}

	max_threads = properties.maxThreadsPerBlock;

	block_size = max_threads / 4;
	grid_size = (int) ((n + m) / block_size) + 1;

	cublasStatus stat;

	stat = cublasInit();

	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		//printf("ERROR: %d", stat);
		return;
	}

	int lenArr = std::max(m, n);
	cublasAlloc(m, sizeof(float), (void**) &d_b);
	cublasAlloc(m + n, sizeof(float), (void**) &d_xe);
	cublasAlloc(m + n, sizeof(float), (void**) &d_old_xe);
	cublasAlloc(m*n, sizeof(float), (void**) &d_A);
	cublasAlloc(lenArr, sizeof(float), (void**) &d_temp);
	cublasAlloc(lenArr, sizeof(float), (void**) &d_temp1);
	cublasAlloc(lenArr, sizeof(float), (void**) &d_temp2);
	cublasAlloc(n, sizeof(float), (void**) &d_z);
	cublasAlloc(n*n, sizeof(float), (void**) &d_G);
	cublasAlloc(m, sizeof(float), (void**) &d_lambda);
	cublasAlloc(m, sizeof(float), (void**) &d_lambda_scaled);
	cublasAlloc(n, sizeof(float), (void**) &d_x_old_apg);
	cublasAlloc(n + m, sizeof(bool), (void**) &d_nz_x);
	cublasAlloc(n + m, sizeof(bool), (void**) &d_nz_x_prev);
	cublasAlloc(n+m, sizeof(float), (void**) &d_xG);
	cublasAlloc(n, sizeof(float), (void**) &d_Gx);
	cublasAlloc(n, sizeof(float), (void**) &d_Gz);
	cublasAlloc(n, sizeof(float), (void**) &d_s);
	cublasAlloc(n, sizeof(float), (void**) &d_Gx_old);

	cublasAlloc(m, sizeof(float), (void**) &dual_temp);
	d_x = d_xe;
	d_e = d_xe + n;

	eps = FLT_EPSILON;
}

void L1Solver_e1x1::free_memory()
{


	cublasFree(d_b);
	cublasFree(d_xe);
	cublasFree(d_old_xe);
	cublasFree(d_A);
	cublasFree(d_temp);
	cublasFree(d_temp1);
	cublasFree(d_temp2);
	cublasFree(d_z);
	cublasFree(d_G);
	cublasFree(d_lambda);
	cublasFree(d_lambda_scaled);
	cublasFree(d_x_old_apg);
	cublasFree(d_nz_x);
	cublasFree(d_nz_x_prev);
	cublasFree(d_xG);
	cublasFree(d_Gx);
	cublasFree(d_Gz);
	cublasFree(d_s);
	cublasFree(d_Gx_old);

	cublasShutdown();
}
void L1Solver_e1x1::set_logFile(FILE* input_file)
{
	log_file = input_file;
	// nIter, nIter_alt_total,norm(diff(x)),norm(diff(x))/norm(prev_x),residual,residual/b,f,eps(diff(f))
	fprintf(log_file, "nIter,n_iter_alt,norm(diff(x)),eps(diff(x)),residual,eps(residual),f,eps(diff(f))\n");
}
void L1Solver_e1x1::set_A(float *A)
{

	// Upload A to the GPU
	cublasSetMatrix(m, n, sizeof(float), A, m, d_A, m);

	// Compute G = A'*A on the cpu and upload to the gpu;
	// % tau = eigs(G,1)*1.1 ;
	// tau = (svds(A,1))^2*1.1 ;
	cublasSgemm('T', 'N', n, n, m, (float)1.0, d_A, m, d_A, m, (float)0.0, d_G, n);
	//cublasSetMatrix(n, n, sizeof(float), G, n, d_G, n);
	tau = cublasSnrm2(m*n, d_A, 1);
	tau = tau*tau;
	// Compute tau from eigenvalues of G.
	//char s = 'S';
	//ssyevx('N', 'I', 'U', n, G, n, 0.0, 0.0, n, n, &s, &tau);
	printf("tau is %f\n", tau);
	//tau = tau;
}
void L1Solver_e1x1::solve(float *b, float *x, float *e, float tol, float tol_int, int maxIter, int maxIter_alt, int stoppingCriterion, const float *xG,float tol2)
{
	

	int ldA = m;
	int ldG = n;

	//maxIter = 50;
	//maxIter_alt = 50;

	tauInv = 1.0 / tau;

	bool verbose = false;

	enum stoppingCriteria stop;
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
	if (stop == STOPPING_GROUND_TRUTH || stop == STOPPING_GROUND_OBJECT)
	{
		cublasSetVector(n+m, sizeof(float), xG, 1, d_xG, 1);
	}
	if (stop == STOPPING_GROUND_OBJECT)
	{
		prev_f = cublasSasum(m+ n, d_xG, 1);
	}
	if (stop == STOPPING_LOG&&log_file==nullptr)
	{
		printf(" log file needed \n");
		exit(1);
	}
	cublasSetVector(m, sizeof(float), b, 1, d_b, 1);
	nrm_b = cublasSnrm2(m, d_b, 1);
	// tol = 5e-2;
	// tol_int = 1e-6 ;
	//tol_int = 1e-6;

	// nIter = 0 ;
	// mu = 2 *m / norm(y,1);
	nIter = 0;
	nIter_alt_total = 0;
	mu = 2.0 * m / cublasSasum(m, d_b, 1);

	// lambda = zeros(m,1);
	// x = zeros(n,1) ;
	// e = y ;
	cudaMemset(d_lambda, 0, m*sizeof(float));
	cudaMemset(d_temp, 0, m*sizeof(float));
	cudaMemset(d_Gx, 0, n*sizeof(float));
	cudaMemset(d_x, 0, n*sizeof(float));
	cublasScopy(m, d_b, 1, d_e, 1);

	// converged_main = 0 ;
	converged_main = false;

	while (!converged_main)
	{
		// while ~converged_main
		//     muInv = 1/mu ;
		//     lambdaScaled = muInv*lambda ;
		muInv = 1.0 / mu;
		
		
		//scale_array << <grid_size, block_size >> >(d_lambda, m, muInv, d_lambda_scaled);
		cuda_scale_array(grid_size, block_size, d_lambda, m, muInv, d_lambda_scaled);

		//     nIter = nIter + 1 ;
		//     e_old_main = e ;
		//     x_old_main = x ;
		nIter += 1;
		cublasScopy(n + m, d_xe, 1, d_old_xe, 1);

		//     temp2 = y + lambdaScaled ;
		//     temp = temp2 - A*x ;
		
		//add_arrays <<<grid_size, block_size >>>(d_b, d_lambda_scaled, d_temp2, m);
		cuda_add_arrays(grid_size, block_size, d_b, d_lambda_scaled, d_temp2, m);
		cublasSaxpy(m, 1, d_temp2, 1, d_temp, 1);
		//	cublasScopy(m, d_temp2, 1, d_temp, 1);
		//	cublasSgemv('N', m, n, -1, d_A, ldA, d_x, 1, 1, d_temp, 1);

		//     e = sign(temp) .* max(abs(temp) - muInv, 0);	
		
		//shrink << <grid_size, block_size >> >(d_temp, muInv, d_e, m);
		cuda_shrink(grid_size, block_size, d_temp, muInv, d_e, m);
		
		//     converged_apg = 0 ;
		//     temp1 = A'*(e - temp2) ;
		converged_apg = false;
		cublasSaxpy(m, -1, d_e, 1, d_temp2, 1);
		cublasSgemv('T', m, n, -1, d_A, ldA, d_temp2, 1, 0, d_temp1, 1);

		//     nIter_apg = 0 ;
		//     t1 = 1 ; z = x ;
		//     muTauInv = muInv*tauInv ;
		nIter_apg = 0;
		t1 = 1;
		cublasScopy(n, d_x, 1, d_z, 1);
		muTauInv = muInv*tauInv;

		//		Gx = G * x;
		//	    Gz = Gx;
		//		cublasSgemv('N', n, n, 1, d_G, ldG, d_x, 1, 0, d_Gx, 1);
		cublasScopy(n, d_Gx, 1, d_Gz, 1);

		while (!converged_apg)
		{
			//         nIter_apg = nIter_apg + 1 ;
			//         x_old_apg = x ;
			//			Gx_old = Gx;
			nIter_apg = nIter_apg + 1;
			cublasScopy(n, d_x, 1, d_x_old_apg, 1);
			cublasScopy(n, d_Gx, 1, d_Gx_old, 1);

			//         temp = z - tauInv*(temp1 + Gz) ;
			
			//scale_sub_array << <grid_size, block_size >> >(d_z, tauInv, d_temp1, d_Gz, d_temp, n);
			cuda_scale_sub_array(grid_size, block_size, d_z, tauInv, d_temp1, d_Gz, d_temp, n);
			//         x = shrink(temp, muTauInv) ;
			//shrink << <grid_size, block_size >> >(d_temp, muTauInv, d_x, n);
			cuda_shrink(grid_size, block_size, d_temp, muTauInv, d_x, n);
			//			Gx = G*x;
			cublasSgemv('N', n, n, 1, d_G, ldG, d_x, 1, 0, d_Gx, 1);

			//		s = tau * (z - x) + Gx - Gz;
			
			//manyOps << <grid_size, block_size >> >(tau, d_z, d_x, d_Gx, d_Gz, d_s, n);
			cuda_manyOps(grid_size, block_size, tau, d_z, d_x, d_Gx, d_Gz, d_s, n);

			//        if norm(s) < tol_int * tau * max(1,norm(x))
			//            converged_apg = 1;
			//        end
			if (cublasSnrm2(n, d_s, 1) < tol_int * tau * std::max(static_cast<float>(1.0), cublasSnrm2(n, d_x, 1)))
			{
				converged_apg = true;
			}



			//         if nIter_apg >= maxIter_alt
			//             converged_apg = 1 ;
			//         end
			if (nIter_apg >= maxIter_alt)
			{
				converged_apg = true;
			}

			//         t2 = (1+sqrt(1+4*t1*t1))/2 ;
			//         z = x + ((t1-1)/t2)*(x-x_old_apg) ;
			//		   Gz = Gx + ((t1-1)/t2) * (Gx - Gx_old);
			//         t1 = t2 ;
			t2 = (1 + sqrt(1 + 4 * t1*t1)) / 2;
			
			//calculate_inner_val << <grid_size, block_size >> >(d_x, (t1 - 1) / t2, d_x_old_apg, d_z, n);
			cuda_calculate_inner_val(grid_size, block_size, d_x, (t1 - 1) / t2, d_x_old_apg, d_z, n);
			//calculate_inner_val << <grid_size, block_size >> >(d_Gx, (t1 - 1) / t2, d_Gx_old, d_Gz, n);
			cuda_calculate_inner_val(grid_size, block_size, d_Gx, (t1 - 1) / t2, d_Gx_old, d_Gz, n);
			t1 = t2;

		}

		nIter_alt_total += nIter_apg;

		//	lambda = lambda + mu*(y - A*x - e) ;
		//temp=-Ax
		cublasSgemv('N', m, n, -1, d_A, ldA, d_x, 1, 0, d_temp, 1);
		cublasScopy(m, d_lambda, 1, d_temp2, 1);
		
		//calculate_lambda << <grid_size, block_size >> >(d_temp2, mu, d_b, d_temp, d_e, d_lambda, m);
		cuda_calculate_lambda(grid_size, block_size, d_temp2, mu, d_b, d_temp, d_e, d_lambda, m);

		//     switch stoppingCriterion
		//         case STOPPING_GROUND_TRUTH
		//             if norm(xG-x) < tol
		//                 converged_main = 1 ;
		//             end
		//         case STOPPING_SUBGRADIENT
		//             error('Duality gap is not a valid stopping criterion for ALM.');
		//         case STOPPING_SPARSE_SUPPORT
		//             % compute the stopping criterion based on the change
		//             % of the number of non-zero components of the estimate
		//             nz_x_prev = nz_x;
		//             nz_x = (abs([x; e])>eps*10);
		//             num_nz_x = sum(nz_x(:));
		//             num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
		//             if num_nz_x >= 1
		//                 criterionActiveSet = num_changes_active / num_nz_x;
		//                 converged_main = ~(criterionActiveSet > tol);
		//             end
		//         case STOPPING_OBJECTIVE_VALUE
		//             % compute the stopping criterion based on the relative
		//             % variation of the objective function.
		//             prev_f = f;
		//             f = norm([x ; e],1);
		//             criterionObjective = abs(f-prev_f)/(prev_f);
		//             converged_main =  ~(criterionObjective > tol);
		//         case STOPPING_DUALITY_GAP
		//             error('Duality gap is not a valid stopping criterion for ALM.');
		//         case STOPPING_INCREMENTS
		//             if norm([x_old_main ; e_old_main] - [x ; e]) < tol*norm([x_old_main ; e_old_main])
		//                 converged_main = 1 ;
		//             end
		//         otherwise
		//             error('Undefined stopping criterion.');
		//     end    

		switch (stop)
		{
		case STOPPING_GROUND_TRUTH:
			//        if norm(xG-x) < tol
			//            converged_main = 1 ;
			//        end
			cublasScopy(n, d_xG, 1, d_temp, 1);
			cublasSaxpy(n, -1, d_x, 1, d_temp, 1);
			dx = cublasSnrm2(n, d_temp, 1);
			if (dx < tol)
				converged_main = true;
			break;
		case STOPPING_SUBGRADIENT:
			//printf("Duality gap is not a valid stopping criterion for ALM.");
			break;
		case STOPPING_SPARSE_SUPPORT:
			/*
				% compute the stopping criterion based on the change
				% of the number of non-zero components of the estimate
				nz_x_prev = nz_x;
				nz_x = (abs(x)>eps*10);
				num_nz_x = sum(nz_x(:));
				num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
				if num_nz_x >= 1
				criterionActiveSet = num_changes_active / num_nz_x;
				converged_main = ~(criterionActiveSet > tol);
				end
				*/

			break;
		case STOPPING_OBJECTIVE_VALUE:
			//          prev_f = f;
			//		    f = norm([x ; e],1);
			//          criterionObjective = abs(f-prev_f)/(prev_f);
			//          converged_main =  ~(criterionObjective > tol);
			prev_f = f;
			f = cublasSasum(n + m, d_xe, 1);
			if (fabs(f - prev_f) <= tol * fabs(prev_f))
			{
				converged_main = true;
			}
			if (verbose)
			{
				//printf("prev_f: %20.20f\n", prev_f);
				//printf("f: %20.20f\n", f);
				//printf("abs(f-prev_f): %20.20f\n", fabs(f-prev_f));
			}
			break;
		case STOPPING_DUALITY_GAP:
			//temp=-Ax
			cublasScopy(m, d_temp, 1, dual_temp, 1);
			cublasSaxpy(m, -1, d_e, 1, dual_temp, 1);
			//temp=-Ax-e
			cublasSaxpy(m, 1, d_b, 1, dual_temp, 1);
			//temp=b-Ax-e
			diff_nrm_b = cublasSnrm2(m, dual_temp, 1);
			if (diff_nrm_b / nrm_b < tol)
			{
				converged_main = true;
			}
			//printf("Duality gap is not a valid stopping criterion for ALM.");
			break;
		case STOPPING_INCREMENTS:
			//            if norm([x_old_main ; e_old_main] - [x ; e]) < tol*norm([x_old_main ; e_old_main])
			//                converged_main = 1 ;
			//            end
			//cublasScopy(n + m, d_old_xe, 1, d_temp1, 1);
			//nxo = cublasSnrm2(n, d_old_xe, 1);
			//neo = cublasSnrm2(m, d_old_xe + n, 1);
			//cublasSaxpy(n + m, -1, d_xe, 1, d_temp1, 1);
			//dx = cublasSnrm2(n, d_temp1, 1);
			//de = cublasSnrm2(m, d_temp1 + n, 1);
			//if (dx < tol*nxo && de < tol * neo)
			//	converged_main = true;
			cublasScopy(n + m, d_old_xe, 1, d_temp1, 1);
			nxo = cublasSnrm2(n+m, d_old_xe, 1);
			cublasSaxpy(n + m, -1, d_xe, 1, d_temp1, 1);
			dx = cublasSnrm2(n+m, d_temp1, 1);
			if (dx < tol*nxo)
			{
				converged_main = true;
			}
			break;
		case STOPPING_KKT_DUAL_TOL:
			//temp=-Ax
			cublasScopy(m, d_temp, 1, dual_temp, 1);
			cublasSaxpy(m, -1, d_e, 1, dual_temp, 1);
			//temp=-Ax-e
			cublasSaxpy(m, 1, d_b, 1, dual_temp, 1);
			//temp=b-Ax-e
			diff_nrm_b = cublasSnrm2(m, dual_temp, 1);
			cublasScopy(n + m, d_old_xe, 1, d_temp1, 1);
			nxo = cublasSnrm2(n + m, d_old_xe, 1);
			cublasSaxpy(n + m, -1, d_xe, 1, d_temp1, 1);
			dx = cublasSnrm2(n + m, d_temp1, 1);
			if (diff_nrm_b< tol*nrm_b&&dx<tol2*nxo)
			{
				converged_main = true;
			}
			break;
		case STOPPING_GROUND_OBJECT:
			f = cublasSasum(n + m, d_xe, 1);
			if (fabs(f - prev_f) <= tol *fabs( prev_f))
			{
				converged_main = true;
			}
			if (verbose)
			{
				//printf("prev_f: %20.20f\n", prev_f);
				//printf("f: %20.20f\n", f);
				//printf("abs(f-prev_f): %20.20f\n", fabs(f-prev_f));
			}
			break;
		case STOPPING_LOG:
			// nIter, nIter_alt_total,norm(diff(x)),norm(diff(x))/norm(prev_x),residual,residual/b,f
			cublasScopy(n + m, d_old_xe, 1, d_temp1, 1);
			nxo = cublasSnrm2(n + m, d_old_xe, 1);
			cublasSaxpy(n + m, -1, d_xe, 1, d_temp1, 1);
			dx = cublasSnrm2(n + m, d_temp1, 1);
			cublasScopy(m, d_temp, 1, dual_temp, 1);
			cublasSaxpy(m, -1, d_e, 1, dual_temp, 1);
			//temp=-Ax-e
			cublasSaxpy(m, 1, d_b, 1, dual_temp, 1);
			//temp=b-Ax-e
			diff_nrm_b = cublasSnrm2(m, dual_temp, 1);
			prev_f = f;
			f = cublasSasum(n + m, d_xe, 1);
			fprintf(log_file, "%lf,%lf,%lf,%lf,", static_cast<double>(nIter), static_cast<double>(nIter_alt_total), static_cast<double>(dx), static_cast<double>(dx / nxo));
			fprintf(log_file, "%lf,%lf,%lf,%lf\n", static_cast<double>(diff_nrm_b), static_cast<double>(diff_nrm_b / nrm_b), static_cast<double>(f), static_cast<double>(fabs((prev_f - f) / f)));
			break;
		default:
			//printf("Undefined stopping criterion. Default");
			break;
		}

	/*	if (!converged_main && dx < 100 * FLT_EPSILON)
		{
			converged_main = true;
		}*/

		//     if ~converged_main && nIter >= maxIter
		//         % disp('Maximum Iterations Reached') ;
		//         converged_main = 1 ;
		// 
		//     end
		if ((!converged_main) && (nIter >= maxIter))
		{
			printf("max iteration %d exceeded\n", maxIter);
			converged_main = true;
		}
	}
	fprintf(stdout, " finished at  %d inner iter and %d out iter\n",nIter_alt_total, nIter);
	cublasGetVector(n, sizeof(float), d_x, 1, x, 1);
	cublasGetVector(m, sizeof(float), d_e, 1, e, 1);
}


