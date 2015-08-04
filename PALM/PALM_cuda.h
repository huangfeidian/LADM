/*******************************************************************
 *
 *   This is a C++ interface to our CUDA implementation of the PALM_CBM L1 solver routine.
 *
 *******************************************************************/
#ifndef __OPTIMIZATION_CUDA_CPP_H__
#define __OPTIMIZATION_CUDA_CPP_H__ 



//#define NULL 0

enum stoppingCriteria{
	 STOPPING_GROUND_TRUTH   = -1,
	 STOPPING_DUALITY_GAP    = 1,
	 STOPPING_SPARSE_SUPPORT = 2,
	 STOPPING_OBJECTIVE_VALUE = 3,
	 STOPPING_SUBGRADIENT    = 4,
	 STOPPING_INCREMENTS     = 5,
	 STOPPING_DEFAULT        = STOPPING_INCREMENTS
};

class L1Solver_e1x1
{
public:
	//      Allocates space to hold A with the correct storage on CPU and on GPU,
	//	and allocates space for other state vectors, etc... on the GPU.
	L1Solver_e1x1(int new_m, int new_n);

	int m,n;
	int nIter;
	int nIter_alt_total;

	//	Clears allocated memory on both CPU side and GPU side.
	~L1Solver_e1x1();

	// Upload the A matrix to GPU memory.
	// A must have close packed column-major storage.
	void set_A(float* A);

	//	Solves the L1-minimization problem.
	//
	//  Inputs:
	//     y -- m by 1. The current test sample.
	// 
	//  Outputs: 
	//     xDest -- the CPU side destination array for x. (n by 1)
	//     eDest -- the CPU side destination array for e. (m by 1)
	//
	//        The input and output arrays can be whatever type is convenient for the rest of the code;
	//        they will likely be copied and typeconverted internally.
	// 
	void solve( float *b, float *x, float *e, float tol=1E-6, float tol_int=1E-6, int maxIter=50, int maxIter_alt=50, int stoppingcriterion=STOPPING_INCREMENTS, const float *xG=NULL); 

protected:

	float tau, tauInv;
	float *d_b, *d_xe, *d_old_xe, *d_A, *d_temp, *d_temp1, *d_temp2, *d_x, *d_e, *d_z, *d_G, *d_Gx, *d_Gz, *d_s, *d_Gx_old;
	float *d_lambda, *d_lambda_scaled, *d_x_old_apg, *d_xG;
	bool  *d_nz_x, *d_nz_x_prev;
	int block_size, grid_size;

	float *G;

	float eps;

};



#endif // __OPTIMIZATION_CUDA_CPP_H_
