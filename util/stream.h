#ifndef __H_stream_H__
#define __H_stream_H__
#include "enum.h"
#include "flags.h"
#include "util.h"
#include <vector>
#include <stdint.h>
namespace alsm
{
	template<DeviceType D>
	struct stream
	{
		int device_index;
		// this is only a dummy implementation for CPU
		stream()
		{
			device_index = -1;
		}
		void sync()
		{

		}
		void destory()
		{

		}
		void set_context()
		{

		}
		static std::vector<stream<D>> create_streams(int stream_size,bool multi_device_support=false)
		{
			std::vector<stream<D>> result_streams(stream_size, stream<D>());
			return result_streams;
		}
	};
#if ALSM_USE_GPU
	template<>
	struct stream< DeviceType::GPU >
	{
		int device_index;
		cudaStream_t cudastream;
		cublasHandle_t local_handle;

		__DEVICE__ stream() 
		{
			device_index = -1;
		}
		stream(const stream<DeviceType::GPU>& in_stream)
		{
			cudastream = in_stream.cudastream;
			local_handle = in_stream.local_handle;
			device_index = in_stream.device_index;
		}
		stream<DeviceType::GPU>& operator=(const stream<DeviceType::GPU>& in_stream)
		{
			if (this != &in_stream)
			{
				cudastream = in_stream.cudastream;
				local_handle = in_stream.local_handle;
				device_index = in_stream.device_index;
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
			CUDA_CHECK_ERR(cudaStreamDestroy(cudastream));
		}
		__DEVICE__ void set_context()
		{
			CUDA_CHECK_ERR(cudaSetDevice(device_index));
		}
		static __DEVICE__ std::vector<stream<DeviceType::GPU>> create_streams(int stream_size,bool multi_device_support=false)
		{
			std::vector<stream<DeviceType::GPU>> result_streams(stream_size, stream<DeviceType::GPU>());
			int gpu_numbers = 0;
			CUDA_CHECK_ERR(cudaGetDeviceCount(&gpu_numbers));
			if (multi_device_support == false)
			{
				int max_mem_device=0;
				uint64_t cuda_mem_size = 0;
				cudaDeviceProp properties;
				for (int i = 0; i < gpu_numbers; i++)
				{
					CUDA_CHECK_ERR(cudaGetDeviceProperties(&properties, i));
					if (cuda_mem_size < properties.totalGlobalMem)
					{
						cuda_mem_size = properties.totalGlobalMem;
						max_mem_device = i;
					}
				}
				CUDA_CHECK_ERR(cudaSetDevice(max_mem_device));
				for (int i = 0; i < stream_size; i++)
				{
					cudaStream_t temp_stream;
					CUDA_CHECK_ERR(cudaStreamCreate(&temp_stream));
					result_streams[i] = stream<DeviceType::GPU>(temp_stream);
					result_streams[i].device_index = max_mem_device;
				}
				return result_streams;
			}
			else
			{
				for (int i = 0; i < stream_size; i++)
				{
					cudaStream_t temp_stream;
					CUDA_CHECK_ERR(cudaSetDevice(i%gpu_numbers));
					CUDA_CHECK_ERR(cudaStreamCreate(&temp_stream));
					result_streams[i] = stream<DeviceType::GPU>(temp_stream);
					result_streams[i].device_index = i%gpu_numbers;
				}
				return result_streams;
			}
		}
		__DEVICE__ ~stream()
		{
			//CUBLAS_CHECK_ERR(cublasDestroy(local_handle));
		}
	};

#endif
	
}
#endif
