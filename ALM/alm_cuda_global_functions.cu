#include <cuda_runtime.h>
#include <math.h>
//------------------ Begin CUDA kernel definitions ---------------------
// shrink(X, alpha)
// Y = sign(X) .* max(abs(X)-alpha, 0);
//float 
__global__ void Sshrink(const float *x, float alpha, float *y, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		y[i] = (fabs(x[i]) - alpha) > 0.0 ? (fabs(x[i]) - alpha) : 0.0;
		//y[i] = std::max(fabs(x[i]) - alpha, static_cast<float>(0.0));
		if (x[i] < 0)
		{
			y[i] = -1 * y[i];
		}
	}
}
void cudaS_shrink(int grid_size, int block_size, const float *x, float alpha, float *y, int len)
{
	Sshrink << <grid_size, block_size >> >(x, alpha, y, len);
}
//__global__ void set_index(float *b, int i, float c)
//{
//	b[i] = c;
//}
//__global__ void set_index(int *b, int i, int c)
//{
//	b[i] = c;
//}

__global__ void Sset_array(float *b, int len, float c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		b[i] = c;
	}
}
void cudaS_set_array(int grid_size, int block_size, float *b, int len, float c)
{
	Sset_array << <grid_size, block_size >> >(b, len, c);
}
// nz_x = (abs(x)>eps*10);
__global__ void Sset_nzx(bool *nz_x, const float *x, int len, float threshold)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		nz_x[i] = fabs(x[i]) > threshold;
	}
}
void cudaS_set_nzx(int grid_size, int block_size, bool *nz_x, const  float *x, int len, float threshold)
{
	Sset_nzx << <grid_size, block_size >> >(nz_x, x, len, threshold);
}

// lambdaScaled = muInv*lambda ;
__global__ void Sscale_array(const float *arr, int len, float a, float *rtn)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = arr[i] * a;
	}
}
void cudaS_scale_array(int grid_size, int block_size, const float *arr, int len, float a, float *rtn)
{
	Sscale_array << <grid_size, block_size >> >(arr, len, a, rtn);
}

// b + lambdaScaled;
__global__ void Sadd_arrays(const float *a, const  float *b, float *c, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		c[i] = a[i] + b[i];
	}
}
void cudaS_add_arrays(int grid_size, int block_size, const  float *a, const float *b, float *c, int len)
{
	Sadd_arrays << <grid_size, block_size >> >(a, b, c, len);
}

//         z = x + ((t1-1)/t2)*(x-x_old_apg) ;
//			calculate_inner_val<<<grid_size, block_size>>>(d_x, , (t1 - 1) / t2, d_x_old_apg, d_z, n);
__global__ void Scalculate_inner_val(const float *x, float a, const float *x_old_apg, float *z, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		z[i] = x[i] + a * (x[i] - x_old_apg[i]);
	}
}
void cudaS_calculate_inner_val(int grid_size, int block_size, const float *x, float a, const float *x_old_apg, float *z, int len)
{
	Scalculate_inner_val << <grid_size, block_size >> >(x, a, x_old_apg, z, len);
}

// scale_sub_array<<<grid_size, block_size>>>(d_z, tauInv, d_temp1, d_Gz, d_temp, n);
// temp1 = z - tauInv*(temp1 + Gz) ;
__global__ void Sscale_sub_array(const float *z, float tauInv, const float *temp1, const float *Gz, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = z[i] - tauInv * (temp1[i] + Gz[i]);
	}
}
void cudaS_scale_sub_array(int grid_size, int block_size, const float *z, float tauInv, const float *temp1, const float *Gz, float *rtn, int len)
{
	Sscale_sub_array << <grid_size, block_size >> >(z, tauInv, temp1, Gz, rtn, len);
}

// manyOps<<<grid_size, block_size>>>(tau, d_z, d_x, d_Gx, d_Gz, d_s, n);
//		s = tau * (z - x) + Gx - Gz;

__global__ void SmanyOps(float tau, const float *z, const float *x, const float *Gx, const float *Gz, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = tau * (z[i] - x[i]) + Gx[i] - Gz[i];
	}
}
void cudaS_manyOps(int grid_size, int block_size, float tau, const  float *z, const float *x, const float *Gx, const float *Gz, float *rtn, int len)
{
	SmanyOps << <grid_size, block_size >> >(tau, z, x, Gx, Gz, rtn, len);
}
//	lambda = lambda + mu*(y - A*x - e) ;
//		calculate_lambda<<<grid_size,block_size>>>(d_temp2, mu, d_b, d_temp1, d_lambda, d_e, m);
__global__ void Scalculate_lambda(const float *lambda, float mu, const float *b, const float *temp, const float *e, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = lambda[i] + mu * (b[i] + temp[i] - e[i]);
	}
}
void cudaS_calculate_lambda(int grid_size, int block_size, const  float *lambda, float mu, const  float *b, const float *temp, const  float *e, float *rtn, int len)
{
	Scalculate_lambda << <grid_size, block_size >> >(lambda, mu, b, temp, e, rtn, len);
}


//// cublasSgemv('N', m, n, -1, d_A, ldA, d_x, 1, 1, d_temp, 1);
//__global__ void my_sgemvn(int m, int n, float alpha, float *A, int ldA, float *x, int ldx, int b, float *temp, int ldt)
//{
//	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	if (i < m)
//	{
//		float dp = 0;
//		for (int j = 0; j < n; j++)
//		{
//			dp += A[j];
//		}
//	}
//}
//void cuda_my_sgemvn(int grid_size, int block_size, int m, int n, float alpha, float *A, int ldA, float *x, int ldx, int b, float *temp, int ldt)
//{
//	my_sgemvn << <grid_size, block_size >> >(m, n, alpha, A, ldA, x, ldx, b, temp,ldt);
//}
//
//// cublasSnrm2(m, d_old_xe_alt+n, 1)
//__global__ void my_snrm2(int len, float *arr, int ld, float *rtn)
//{
//	*rtn = 0;
//	for (int i = 0; i < len; i++)
//	{
//		*rtn += arr[i * ld] * arr[i * ld];
//	}
//}
//void cuda_my_snrm2(int grid_size,int block_size,int len, float *arr, int ld, float *rtn)
//{
//	my_snrm2 << <grid_size, block_size >> >(len, arr, ld, rtn);
//}

// beta * (temp - z) + x
__global__ void Sbtzx(float beta, const float *temp, const  float *z, const float *x, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = beta * (temp[i] - z[i]) + x[i];
	}
}
void cudaS_btzx(int grid_size, int block_size, float beta, const float *temp, const float *z, const float *x, float *rtn, int len)
{
	Sbtzx << <grid_size, block_size >> >(beta, temp, z, x, rtn, len);
}
//	    x = x - beta * (z - temp);
__global__ void Sxbzt(const float *x, float b, const float *z, const float *temp, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = x[i] - b * (z[i] - temp[i]);
	}
}
void cudaS_xbzt(int grid_size, int block_size, const float *x, float b, const float *z, const  float *temp, float *rtn, int len)
{
	Sxbzt << <grid_size, block_size >> >(x, b, z, temp, rtn, len);
}
// z = sign(temp1).*min(1,abs(temp1));
__global__ void Ssign_min(const float *temp1, float *z, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		float t = temp1[i];
		if (t == 0)
		{
			z[i] = 0;
		}
		else
		{
			if (t < 0)  z[i] = -1 * min(1.0f, fabs(t));
			else        z[i] = min(1.0f, fabs(t));
			//			z[i]    = t / fabs(t) * min(1.0, fabs(t));
		}
	}
}
void cudaS_sign_min(int grid_size, int block_size, const float *temp1, float *z, int len)
{
	Ssign_min << <grid_size, block_size >> >(temp1, z, len);
}

//double
__global__ void Dshrink(const double *x, double alpha, double *y, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		y[i] = (fabs(x[i]) - alpha) > 0.0 ? (fabs(x[i]) - alpha) : 0.0;
		//y[i] = std::max(fabs(x[i]) - alpha, static_cast<double>(0.0));
		if (x[i] < 0)
		{
			y[i] = -1 * y[i];
		}
	}
}
void cudaD_shrink(int grid_size, int block_size, const double *x, double alpha, double *y, int len)
{
	Dshrink << <grid_size, block_size >> >(x, alpha, y, len);
}
//__global__ void set_index(double *b, int i, double c)
//{
//	b[i] = c;
//}
//__global__ void set_index(int *b, int i, int c)
//{
//	b[i] = c;
//}

__global__ void Dset_array(double *b, int len, double c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		b[i] = c;
	}
}
void cudaD_set_array(int grid_size, int block_size, double *b, int len, double c)
{
	Dset_array << <grid_size, block_size >> >(b, len, c);
}
// nz_x = (abs(x)>eps*10);
__global__ void Dset_nzx(bool *nz_x, const double *x, int len, double threshold)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		nz_x[i] = fabs(x[i]) > threshold;
	}
}
void cudaD_set_nzx(int grid_size, int block_size, bool *nz_x, const  double *x, int len, double threshold)
{
	Dset_nzx << <grid_size, block_size >> >(nz_x, x, len, threshold);
}

// lambdaScaled = muInv*lambda ;
__global__ void Dscale_array(const double *arr, int len, double a, double *rtn)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = arr[i] * a;
	}
}
void cudaD_scale_array(int grid_size, int block_size, const double *arr, int len, double a, double *rtn)
{
	Dscale_array << <grid_size, block_size >> >(arr, len, a, rtn);
}

// b + lambdaScaled;
__global__ void Dadd_arrays(const double *a, const  double *b, double *c, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		c[i] = a[i] + b[i];
	}
}
void cudaD_add_arrays(int grid_size, int block_size, const  double *a, const double *b, double *c, int len)
{
	Dadd_arrays << <grid_size, block_size >> >(a, b, c, len);
}

//         z = x + ((t1-1)/t2)*(x-x_old_apg) ;
//			calculate_inner_val<<<grid_size, block_size>>>(d_x, , (t1 - 1) / t2, d_x_old_apg, d_z, n);
__global__ void Dcalculate_inner_val(const double *x, double a, const double *x_old_apg, double *z, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		z[i] = x[i] + a * (x[i] - x_old_apg[i]);
	}
}
void cudaD_calculate_inner_val(int grid_size, int block_size, const double *x, double a, const double *x_old_apg, double *z, int len)
{
	Dcalculate_inner_val << <grid_size, block_size >> >(x, a, x_old_apg, z, len);
}

// scale_sub_array<<<grid_size, block_size>>>(d_z, tauInv, d_temp1, d_Gz, d_temp, n);
// temp1 = z - tauInv*(temp1 + Gz) ;
__global__ void Dscale_sub_array(const double *z, double tauInv, const double *temp1, const double *Gz, double *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = z[i] - tauInv * (temp1[i] + Gz[i]);
	}
}
void cudaD_scale_sub_array(int grid_size, int block_size, const double *z, double tauInv, const double *temp1, const double *Gz, double *rtn, int len)
{
	Dscale_sub_array << <grid_size, block_size >> >(z, tauInv, temp1, Gz, rtn, len);
}

// manyOps<<<grid_size, block_size>>>(tau, d_z, d_x, d_Gx, d_Gz, d_s, n);
//		s = tau * (z - x) + Gx - Gz;

__global__ void DmanyOps(double tau, const double *z, const double *x, const double *Gx, const double *Gz, double *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = tau * (z[i] - x[i]) + Gx[i] - Gz[i];
	}
}
void cudaD_manyOps(int grid_size, int block_size, double tau, const  double *z, const double *x, const double *Gx, const double *Gz, double *rtn, int len)
{
	DmanyOps << <grid_size, block_size >> >(tau, z, x, Gx, Gz, rtn, len);
}
//	lambda = lambda + mu*(y - A*x - e) ;
//		calculate_lambda<<<grid_size,block_size>>>(d_temp2, mu, d_b, d_temp1, d_lambda, d_e, m);
__global__ void Dcalculate_lambda(const double *lambda, double mu, const double *b, const double *temp, const double *e, double *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = lambda[i] + mu * (b[i] + temp[i] - e[i]);
	}
}
void cudaD_calculate_lambda(int grid_size, int block_size, const  double *lambda, double mu, const  double *b, const double *temp, const  double *e, double *rtn, int len)
{
	Dcalculate_lambda << <grid_size, block_size >> >(lambda, mu, b, temp, e, rtn, len);
}


//// cublasSgemv('N', m, n, -1, d_A, ldA, d_x, 1, 1, d_temp, 1);
//__global__ void my_sgemvn(int m, int n, double alpha, double *A, int ldA, double *x, int ldx, int b, double *temp, int ldt)
//{
//	int i = threadIdx.x + blockDim.x * blockIdx.x;
//	if (i < m)
//	{
//		double dp = 0;
//		for (int j = 0; j < n; j++)
//		{
//			dp += A[j];
//		}
//	}
//}
//void cuda_my_sgemvn(int grid_size, int block_size, int m, int n, double alpha, double *A, int ldA, double *x, int ldx, int b, double *temp, int ldt)
//{
//	my_sgemvn << <grid_size, block_size >> >(m, n, alpha, A, ldA, x, ldx, b, temp,ldt);
//}
//
//// cublasSnrm2(m, d_old_xe_alt+n, 1)
//__global__ void my_snrm2(int len, double *arr, int ld, double *rtn)
//{
//	*rtn = 0;
//	for (int i = 0; i < len; i++)
//	{
//		*rtn += arr[i * ld] * arr[i * ld];
//	}
//}
//void cuda_my_snrm2(int grid_size,int block_size,int len, double *arr, int ld, double *rtn)
//{
//	my_snrm2 << <grid_size, block_size >> >(len, arr, ld, rtn);
//}

// beta * (temp - z) + x
__global__ void Dbtzx(double beta, const double *temp, const  double *z, const double *x, double *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = beta * (temp[i] - z[i]) + x[i];
	}
}
void cudaD_btzx(int grid_size, int block_size, double beta, const double *temp, const double *z, const double *x, double *rtn, int len)
{
	Dbtzx << <grid_size, block_size >> >(beta, temp, z, x, rtn, len);
}
//	    x = x - beta * (z - temp);
__global__ void Dxbzt(const double *x, double b, const double *z, const double *temp, double *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = x[i] - b * (z[i] - temp[i]);
	}
}
void cudaD_xbzt(int grid_size, int block_size, const double *x, double b, const double *z, const  double *temp, double *rtn, int len)
{
	Dxbzt << <grid_size, block_size >> >(x, b, z, temp, rtn, len);
}
// z = sign(temp1).*min(1,abs(temp1));
__global__ void Dsign_min(const double *temp1, double *z, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		double t = temp1[i];
		if (t == 0)
		{
			z[i] = 0;
		}
		else
		{
			if (t < 0)  z[i] = -1 * min(1.0f, fabs(t));
			else        z[i] = min(1.0f, fabs(t));
			//			z[i]    = t / fabs(t) * min(1.0, fabs(t));
		}
	}
}
void cudaD_sign_min(int grid_size, int block_size, const double *temp1, double *z, int len)
{
	Dsign_min << <grid_size, block_size >> >(temp1, z, len);
}