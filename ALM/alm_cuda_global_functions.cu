#include <cuda_runtime.h>
#include <math.h>
//------------------ Begin CUDA kernel definitions ---------------------
// shrink(X, alpha)
// Y = sign(X) .* max(abs(X)-alpha, 0);
__global__ void shrink(const float *x, float alpha, float *y, int len)
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
void cuda_shrink(int grid_size, int block_size, const float *x, float alpha, float *y, int len)
{
	shrink << <grid_size, block_size >> >(x, alpha, y, len);
}
//__global__ void set_index(float *b, int i, float c)
//{
//	b[i] = c;
//}
//__global__ void set_index(int *b, int i, int c)
//{
//	b[i] = c;
//}

__global__ void set_array(float *b, int len, float c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		b[i] = c;
	}
}
void cuda_set_array(int grid_size, int block_size, float *b, int len, float c)
{
	set_array << <grid_size, block_size >> >(b, len, c);
}
// nz_x = (abs(x)>eps*10);
__global__ void set_nzx(bool *nz_x, const float *x, int len, float threshold)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		nz_x[i] = fabs(x[i]) > threshold;
	}
}
void cuda_set_nzx(int grid_size, int block_size, bool *nz_x, const  float *x, int len, float threshold)
{
	set_nzx << <grid_size, block_size >> >(nz_x, x, len, threshold);
}

// lambdaScaled = muInv*lambda ;
__global__ void scale_array(const float *arr, int len, float a, float *rtn)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = arr[i] * a;
	}
}
void cuda_scale_array(int grid_size, int block_size, const float *arr, int len, float a, float *rtn)
{
	scale_array << <grid_size, block_size >> >(arr, len, a, rtn);
}

// b + lambdaScaled;
__global__ void add_arrays(const float *a, const  float *b, float *c, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		c[i] = a[i] + b[i];
	}
}
void cuda_add_arrays(int grid_size, int block_size, const  float *a, const float *b, float *c, int len)
{
	add_arrays << <grid_size, block_size >> >(a, b, c, len);
}

//         z = x + ((t1-1)/t2)*(x-x_old_apg) ;
//			calculate_inner_val<<<grid_size, block_size>>>(d_x, , (t1 - 1) / t2, d_x_old_apg, d_z, n);
__global__ void calculate_inner_val(const float *x, float a, const float *x_old_apg, float *z, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		z[i] = x[i] + a * (x[i] - x_old_apg[i]);
	}
}
void cuda_calculate_inner_val(int grid_size, int block_size, const float *x, float a, const float *x_old_apg, float *z, int len)
{
	calculate_inner_val << <grid_size, block_size >> >(x, a, x_old_apg, z, len);
}

// scale_sub_array<<<grid_size, block_size>>>(d_z, tauInv, d_temp1, d_Gz, d_temp, n);
// temp1 = z - tauInv*(temp1 + Gz) ;
__global__ void scale_sub_array(const float *z, float tauInv, const float *temp1, const float *Gz, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = z[i] - tauInv * (temp1[i] + Gz[i]);
	}
}
void cuda_scale_sub_array(int grid_size, int block_size, const float *z, float tauInv, const float *temp1, const float *Gz, float *rtn, int len)
{
	scale_sub_array << <grid_size, block_size >> >(z, tauInv, temp1, Gz, rtn, len);
}

// manyOps<<<grid_size, block_size>>>(tau, d_z, d_x, d_Gx, d_Gz, d_s, n);
//		s = tau * (z - x) + Gx - Gz;

__global__ void manyOps(float tau, const float *z, const float *x, const float *Gx, const float *Gz, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = tau * (z[i] - x[i]) + Gx[i] - Gz[i];
	}
}
void cuda_manyOps(int grid_size, int block_size, float tau, const  float *z, const float *x, const float *Gx, const float *Gz, float *rtn, int len)
{
	manyOps << <grid_size, block_size >> >(tau, z, x, Gx, Gz, rtn, len);
}
//	lambda = lambda + mu*(y - A*x - e) ;
//		calculate_lambda<<<grid_size,block_size>>>(d_temp2, mu, d_b, d_temp1, d_lambda, d_e, m);
__global__ void calculate_lambda(const float *lambda, float mu, const float *b, const float *temp, const float *e, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = lambda[i] + mu * (b[i] + temp[i] - e[i]);
	}
}
void cuda_calculate_lambda(int grid_size, int block_size, const  float *lambda, float mu, const  float *b, const float *temp, const  float *e, float *rtn, int len)
{
	calculate_lambda << <grid_size, block_size >> >(lambda, mu, b, temp, e, rtn, len);
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
__global__ void btzx(float beta, const float *temp, const  float *z, const float *x, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = beta * (temp[i] - z[i]) + x[i];
	}
}
void cuda_btzx(int grid_size, int block_size, float beta, const float *temp, const float *z, const float *x, float *rtn, int len)
{
	btzx << <grid_size, block_size >> >(beta, temp, z, x, rtn, len);
}
//	    x = x - beta * (z - temp);
__global__ void xbzt(const float *x, float b, const float *z, const float *temp, float *rtn, int len)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < len)
	{
		rtn[i] = x[i] - b * (z[i] - temp[i]);
	}
}
void cuda_xbzt(int grid_size, int block_size, const float *x, float b, const float *z, const  float *temp, float *rtn, int len)
{
	xbzt << <grid_size, block_size >> >(x, b, z, temp, rtn, len);
}
// z = sign(temp1).*min(1,abs(temp1));
__global__ void sign_min(const float *temp1, float *z, int len)
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
void cuda_sign_min(int grid_size, int block_size, const float *temp1, float *z, int len)
{
	sign_min << <grid_size, block_size >> >(temp1, z, len);
}