#ifndef __H_ALLOCA_H__
#define __H_ALLOCA_H__
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
	template <DeviceType D>
	void  device_memset(void* dst, int value, int size);
	template <>void  device_memset<DeviceType::CPU>(void* dst, int value, int size)
	{
		memset(dst, value, size);
	}
#if ALSM_USE_GPU
	template<> void device_memset<DeviceType::GPU>(void* dst, int value, int size)
	{
		CUDA_CHECK_ERR(cudaMemset(dst, value, size));
	}

#endif
	template <DeviceType D,typename T>
	void alsm_memset(T*  dst, int value, int size)
	{
		device_memset<D>(static_cast<void*>(dst), value, size*sizeof(T));
	}
	template<DeviceType D>
	void* aligned_malloc(uint64_t size, uint64_t aligned_size);
	template<>
	void* aligned_malloc<DeviceType::CPU>(uint64_t size, uint64_t aligned_size)
	{
		uint64_t alloca_size = size + aligned_size - 1;
		uint64_t real_adr = reinterpret_cast<uint64_t>(malloc(alloca_size));
		uint64_t aligned_adr = ((real_adr + aligned_size - 1) / aligned_size)*aligned_size;
		//printf("malloc %d byte aligned by %d : the return is %x while the origin is %x\n", size, aligned_size, aligned_adr, real_adr);
		mem_map[aligned_adr] = std::make_pair(DeviceType::CPU, real_adr);
		return reinterpret_cast<void*>(aligned_adr);
	}
#if ALSM_USE_GPU
	template<>
	void* aligned_malloc<DeviceType::GPU>(uint64_t size, uint64_t aligned_size)
	{
		uint64_t alloca_size = size + aligned_size - 1;
		uint64_t real_adr;
		CUDA_CHECK_ERR(cudaMalloc((void**) &real_adr, sizeof(float)*alloca_size));
		uint64_t aligned_adr = ((real_adr + aligned_size - 1) / aligned_size)*aligned_size;
		//printf("malloc %d byte aligned by %d : the return is %x while the origin is %x\n", size, aligned_size, aligned_adr, real_adr);
		mem_map[aligned_adr] = std::make_pair(DeviceType::GPU, real_adr);
		return reinterpret_cast<void*>(aligned_adr);
	}
#endif
	template <DeviceType D, typename T> T* alsm_malloc(size_t typed_memory_size)
	{
		T* return_ptr = reinterpret_cast<T*>(aligned_malloc<D>(sizeof(T)*typed_memory_size, ALIGNBLOCK));
		alsm_memset<D, T>(return_ptr, 0, typed_memory_size);
		return return_ptr;
	}
	template <DeviceType D> void alsm_free(void* input_ptr);
	template<> void alsm_free<DeviceType::CPU>(void* input_ptr)
	{
		void* real_adr = reinterpret_cast<void*>(mem_map[reinterpret_cast<uint64_t>(input_ptr)].second);
		//printf("free aligned block %x while the real address is%x\n", reinterpret_cast<uint64_t>(input_ptr), real_adr);
		free(real_adr);;
		//mem_map.erase(reinterpret_cast<uint64_t>(input_ptr));
	}

#if ALSM_USE_GPU
	template<> void alsm_free<DeviceType::GPU>(void* input_ptr)
	{
		void* real_adr = reinterpret_cast<void*>(mem_map[reinterpret_cast<uint64_t>(input_ptr)].second);
		//printf("free aligned block %x while the real address is%x\n", reinterpret_cast<uint64_t>(input_ptr), real_adr);
		CUDA_CHECK_ERR(cudaFree(real_adr));
		//mem_map.erase(reinterpret_cast<uint64_t>(input_ptr));
	}
#endif
	void alsm_free_all()
	{
		for (auto const& i : mem_map)
		{
			if (i.second.first == DeviceType::CPU)
			{
				//printf("free aligned block %x while the real address is%x\n", i.first, i.second.second);
				alsm_free<DeviceType::CPU>(reinterpret_cast<void*>(i.second.second));

			}
#if ALSM_USE_GPU
			else
			{
				alsm_free<DeviceType::GPU>(reinterpret_cast<void*>(i.second.second));
			}
#endif
		}
		mem_map.clear();
	}

//	template< DeviceType D, typename T> void alsm_memcpy(const stream<D>& stream, T* dst_y, const T* from_x, int n);
//	template<> void alsm_memcpy<DeviceType::CPU, float >(const stream<DeviceType::CPU>& stream, float* dst_y, const float* from_x, int n)
//	{
//		if (dst_y != from_x)
//		{
//			memcpy(dst_y, from_x, sizeof(float)*n);
//		}
//
//
//	}
//	template<> void alsm_memcpy<DeviceType::CPU, double >(const stream<DeviceType::CPU>& stream, double* dst_y, const double* from_x, int n)
//	{
//		if (dst_y != from_x)
//		{
//			memcpy(dst_y, from_x, sizeof(double)*n);
//		}
//
//	}
//#if ALSM_USE_GPU
//	template<> void alsm_memcpy<DeviceType::GPU, float >(const stream<DeviceType::GPU>& stream, float* dst_y, const float* from_x, int n)
//	{
//		if (dst_y != from_x)
//		{
//			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(float)*n, cudaMemcpyDeviceToDevice));
//		}
//
//	}
//	template<> void alsm_memcpy<DeviceType::GPU, double >(const stream<DeviceType::GPU>& stream, double* dst_y, const double* from_x, int n)
//	{
//		if (dst_y != from_x)
//		{
//			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(double)*n, cudaMemcpyDeviceToDevice));
//		}
//
//	}
//#endif
	template< DeviceType D, typename T> 
	void alsm_tocpu(const stream<D>& stream, T* dst_y, const T* from_x, int n);
	template<> 
	void alsm_tocpu<DeviceType::CPU, float >(const stream<DeviceType::CPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(float)*n);
		}


	}
	template<> 
	void alsm_tocpu<DeviceType::CPU, double >(const stream<DeviceType::CPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(double)*n);
		}

	}
#if ALSM_USE_GPU
	template<> 
	void alsm_tocpu<DeviceType::GPU, float >(const stream<DeviceType::GPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(float)*n, cudaMemcpyDeviceToHost));
		}

	}
	template<> 
	void alsm_tocpu<DeviceType::GPU, double >(const stream<DeviceType::GPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(double)*n, cudaMemcpyDeviceToHost));
		}
	}
#endif
	template< DeviceType D, typename T> 
	void alsm_fromcpu(const stream<D>& stream, T* dst_y, const T* from_x, int n);
	template<> 
	void alsm_fromcpu<DeviceType::CPU, float >(const stream<DeviceType::CPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(float)*n);
		}


	}
	template<> 
	void alsm_fromcpu<DeviceType::CPU, double >(const stream<DeviceType::CPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(double)*n);
		}

	}
#if ALSM_USE_GPU
	template<> 
	void alsm_fromcpu<DeviceType::GPU, float >(const stream<DeviceType::GPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(float)*n, cudaMemcpyHostToDevice));
		}

	}
	template<> 
	void alsm_fromcpu<DeviceType::GPU, double >(const stream<DeviceType::GPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(double)*n, cudaMemcpyHostToDevice));
		}
	}
#endif
}
#endif
