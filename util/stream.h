#ifndef __H_stream_H__
#define __H_stream_H__
#include "enum.h"
#include "flags.h"
#include "util.h"
namespace alsm
{
	template<DeviceType D>
	struct stream
	{
		// this is only a dummy implementation for CPU
		void sync()
		{

		}
	};
#if ALSM_USE_GPU
	template<>
	struct stream< DeviceType::GPU >
	{
		cudaStream_t cudastream;
		cublasHandle_t local_handle;

		__DEVICE__ stream() 
		{
			
		}
		stream<DeviceType::GPU>& operator=(const stream<DeviceType::GPU>& in_stream)
		{
			if (this != &in_stream)
			{
				cudastream = in_stream.cudastream;
				local_handle = in_stream.local_handle;
			}
			return *this;
		}
		__DEVICE__ stream(cudaStream_t in_stream) :cudastream(in_stream)
		{
			CUBLAS_CHECK_ERR(cublasCreate(&local_handle));
			CUBLAS_CHECK_ERR(cublasSetStream(local_handle, cudastream));
		}
		__DEVICE__ void sync()
		{
			CUDA_CHECK_ERR(cudaStreamSynchronize(cudastream));
		}
		__DEVICE__ ~stream()
		{
			//CUBLAS_CHECK_ERR(cublasDestroy(local_handle));
		}
	};
#endif
}
#endif
