#ifndef __H_stream_H__
#define __H_stream_H__
#include "enum.h"
#include "flags.h"
#include "util.h"
#include <vector>
namespace alsm
{
	template<DeviceType D>
	struct stream
	{
		// this is only a dummy implementation for CPU
		void sync()
		{

		}
		void destory()
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
		stream(const stream<DeviceType::GPU>& in_stream)
		{
			cudastream = in_stream.cudastream;
			local_handle = in_stream.local_handle;
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
		__DEVICE__ void destory()
		{
			CUBLAS_CHECK_ERR(cublasDestroy(local_handle));
			cuStreamDestroy(cudastream);
		}
		__DEVICE__ ~stream()
		{
			//CUBLAS_CHECK_ERR(cublasDestroy(local_handle));
		}
	};
	/*template <DeviceType D>*/
	/*std::vector<stream<D>> create_streams(int stream_size);*/
	//template <>
	//std::vector<stream<DeviceType::CPU>> create_streams<DeviceType::CPU>(int stream_size)
	//{
	//	std::vector<stream<DeviceType::CPU>> result_streams(stream_size, stream<DeviceType::CPU>());
	//	return result_streams;
	//}
	//template <>
	//std::vector<stream<DeviceType::GPU>> create_streams<DeviceType::GPU>(int stream_size)
	//{
	//	std::vector<stream<DeviceType::GPU>> result_streams(stream_size, stream<DeviceType::GPU>());
	//	for (int i = 0; i < stream_size; i++)
	//	{
	//		cudaStream_t temp_stream;
	//		CUDA_CHECK_ERR(cudaStreamCreate(&temp_stream));
	//		result_streams[i] = stream<DeviceType::GPU>(temp_stream);
	//	}
	//	return result_streams;
	//}
#endif
}
#endif
