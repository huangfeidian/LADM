#ifndef __H_alloca_H__
#define __H_alloca_H__
#include "../util/flags.h"
#include "../util/enum.h"
#include "../util/util.h"
#include "../util/stream.h"

#include <map>
#include <stdint.h>




#define ALIGNBLOCK 64
namespace alsm
{
	static std::map<uint64_t, std::pair<DeviceType, uint64_t>> mem_map;
	void* aligned_malloc(uint64_t size, uint64_t aligned_size)
	{
		uint64_t alloca_size = size + aligned_size - 1;
		uint64_t real_adr = reinterpret_cast<uint64_t>(malloc(alloca_size));
		uint64_t aligned_adr = ((real_adr + aligned_size - 1) / aligned_size)*aligned_size;
		printf("malloc %d byte aligned by %d : the return is %x while the origin is %x\n", size, aligned_size, aligned_adr, real_adr);
		mem_map[aligned_adr] = std::make_pair(DeviceType::CPU, real_adr);
		return reinterpret_cast<void*>(aligned_adr);
	}
	template <DeviceType D, typename T> T* alsm_malloc(size_t memory_size);
	template<> double* alsm_malloc<DeviceType::CPU, double>(size_t memory_size)
	{
		double* return_ptr = reinterpret_cast<double*>(aligned_malloc(sizeof(double)*memory_size, ALIGNBLOCK));
		return return_ptr;
	}
	template<> float* alsm_malloc<DeviceType::CPU, float>(size_t memory_size)
	{
		float* return_ptr = reinterpret_cast<float*>(aligned_malloc(sizeof(float)*memory_size, ALIGNBLOCK));
		return return_ptr;
	}
#if ALSM_USE_CUDA
	template<> float* alsm_malloc<DeviceType::GPU, float>(size_t memory_size)
	{
		float* return_ptr;
		CUDA_CHECK_ERR(cudaMalloc((void**) &return_ptr, sizeof(float)*memory_size));
		mem_map[reinterpret_cast<uint64_t>(return_ptr)] = std::make_pair(DeviceType::GPU, reinterpret_cast<uint64_t>(return_ptr));
		return static_cast<float*>(return_ptr);
	}
	template<> double* alsm_malloc<DeviceType::GPU, double>(size_t memory_size)
	{
		double* return_ptr;
		CUDA_CHECK_ERR(cudaMalloc((void**) &return_ptr, sizeof(double)*memory_size));
		mem_map[reinterpret_cast<uint64_t>(return_ptr)] = std::make_pair(DeviceType::GPU, reinterpret_cast<uint64_t>(return_ptr));
		return static_cast<double*>(return_ptr);
	}
#endif
	template <DeviceType D, typename T>
	void alsm_free(T* input_ptr);
	template<> void alsm_free<DeviceType::CPU, float>(float* input_ptr)
	{
		void* real_adr = reinterpret_cast<void*>(mem_map[reinterpret_cast<uint64_t>(input_ptr)].second);
		printf("free aligned block %x while the real address is%x\n", reinterpret_cast<uint64_t>(input_ptr), real_adr);
		free(real_adr);;
		mem_map.erase(reinterpret_cast<uint64_t>(input_ptr));
	}
	template<> void alsm_free<DeviceType::CPU, double>(double* input_ptr)
	{
		void* real_adr = reinterpret_cast<void*>(mem_map[reinterpret_cast<uint64_t>(input_ptr)].second);
		printf("free aligned block %x while the real address is%x\n", reinterpret_cast<uint64_t>(input_ptr), real_adr);
		free(real_adr);;
		mem_map.erase(reinterpret_cast<uint64_t>(input_ptr));
	}
#if ALSM_USE_CUDA
	template<> void alsm_free<DeviceType::GPU, float>(float* input_ptr)
	{
		CUDA_CHECK_ERR(cudaFree(input_ptr));
		mem_map.erase(reinterpret_cast<uint64_t>(input_ptr));
	}
	template<> void alsm_free<DeviceType::GPU, double>(double* input_ptr)
	{
		CUDA_CHECK_ERR(cudaFree(input_ptr));
		mem_map.erase(reinterpret_cast<uint64_t>(input_ptr));
	}
#endif
	void alsm_free_all()
	{
		for (auto const& i : mem_map)
		{
			if (i.second.first == DeviceType::CPU)
			{
				printf("free aligned block %x while the real address is%x\n", i.first, i.second.second);
				free(reinterpret_cast<void*>(i.second.second));

			}
#if ALSM_USE_CUDA
			else
			{
				CUDA_CHECK_ERR(cudaFree(reinterpret_cast<void*>(i.first)));
			}
#endif
		}
		mem_map.empty();
	}
	template <DeviceType D, typename T>
	void  alsm_memset(T* dst, int value, int size);
	template <>void  alsm_memset<DeviceType::CPU, float>(float* dst, int value, int size)
	{
		memset(dst, value, size*sizeof(float));
	}
	template <>void  alsm_memset<DeviceType::CPU, double>(double* dst, int value, int size)
	{
		memset(dst, value, size*sizeof(double));
	}
#if ALSM_USE_CUDA
	template<> void alsm_memset<DeviceType::GPU, float>(float* dst, int value, int size)
	{
		CUDA_CHECK_ERR(cudaMemset(dst, value, size*sizeof(float)));
	}
	template<> void alsm_memset<DeviceType::GPU, double>(double* dst, int value, int size)
	{
		CUDA_CHECK_ERR(cudaMemset(dst, value, size*sizeof(double)));
	}
#endif
	template< DeviceType D, typename T> void alsm_memcpy(const stream<D>& stream, T* dst_y, const T* from_x, int n);
	template<> void alsm_memcpy<DeviceType::CPU, float >(const stream<DeviceType::CPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(float)*n);
		}


	}
	template<> void alsm_memcpy<DeviceType::CPU, double >(const stream<DeviceType::CPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(double)*n);
		}

	}
#if ALSM_USE_CUDA
	template<> void alsm_memcpy<DeviceType::GPU, float >(const stream<DeviceType::GPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(float)*n, cudaMemcpyDeviceToDevice));
		}

	}
	template<> void alsm_memcpy<DeviceType::GPU, double >(const stream<DeviceType::GPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(double)*n, cudaMemcpyDeviceToDevice));
		}

	}
#endif
}
#endif
