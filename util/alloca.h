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
	static std::map<uint64_t, std::pair<int, uint64_t>> mem_map;
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
	void* aligned_malloc(stream<D> stream, uint64_t size, uint64_t aligned_size);
	template<>
	void* aligned_malloc<DeviceType::CPU>(stream<DeviceType::CPU> stream, uint64_t size, uint64_t aligned_size)
	{
		uint64_t alloca_size = size + aligned_size - 1;
		uint64_t real_adr = reinterpret_cast<uint64_t>(malloc(alloca_size));
		uint64_t aligned_adr = ((real_adr + aligned_size - 1) / aligned_size)*aligned_size;
		//printf("malloc %d byte aligned by %d : the return is %x while the origin is %x\n", size, aligned_size, aligned_adr, real_adr);
		mem_map[aligned_adr] = std::make_pair(-1, real_adr);
		return reinterpret_cast<void*>(aligned_adr);
	}
#if ALSM_USE_GPU
	template<>
	void* aligned_malloc<DeviceType::GPU>(stream<DeviceType::GPU> stream,uint64_t size, uint64_t aligned_size)
	{
		stream.set_context();
		uint64_t alloca_size = size + aligned_size - 1;
		uint64_t real_adr;
		cudaSetDevice(stream.device_index);
		CUDA_CHECK_ERR(cudaMalloc((void**) &real_adr, sizeof(float)*alloca_size));
		uint64_t aligned_adr = ((real_adr + aligned_size - 1) / aligned_size)*aligned_size;
		//printf("malloc %d byte aligned by %d : the return is %x while the origin is %x\n", size, aligned_size, aligned_adr, real_adr);
		mem_map[aligned_adr] = std::make_pair(stream.device_index, real_adr);
		return reinterpret_cast<void*>(aligned_adr);
	}
#endif
	template <DeviceType D, typename T> T* alsm_malloc(stream<D> stream,size_t typed_memory_size)
	{
		T* return_ptr = reinterpret_cast<T*>(aligned_malloc<D>(stream,sizeof(T)*typed_memory_size, ALIGNBLOCK));
		alsm_memset<D, T>(return_ptr, 0, typed_memory_size);
		return return_ptr;
	}
	template <DeviceType D> void alsm_free(void* input_ptr);
	template<> void alsm_free<DeviceType::CPU>(void* input_ptr)
	{
		void* real_adr = reinterpret_cast<void*>(mem_map[reinterpret_cast<uint64_t>(input_ptr)].second);
		free(real_adr);
		mem_map.erase(reinterpret_cast<uint64_t>(input_ptr));
	}

#if ALSM_USE_GPU
	template<> void alsm_free<DeviceType::GPU>(void* input_ptr)
	{
		CUDA_CHECK_ERR(cudaSetDevice(mem_map[reinterpret_cast<uint64_t>(input_ptr)].first));
		void* real_adr = reinterpret_cast<void*>(mem_map[reinterpret_cast<uint64_t>(input_ptr)].second);

		CUDA_CHECK_ERR(cudaFree(real_adr));
		//mem_map.erase(reinterpret_cast<uint64_t>(input_ptr));
	}
#endif
	void alsm_free_all()
	{
		for (auto& i : mem_map)
		{
			if (i.second.first == -1)
			{
				//printf("free aligned block %x while the real address is%x\n", i.first, i.second.second);
				void* real_adr = reinterpret_cast<void*>(i.second.second);
				free(real_adr);
				i.second.second = 0;

			}
#if ALSM_USE_GPU
			else
			{
				CUDA_CHECK_ERR(cudaSetDevice(i.second.first));
				void* real_adr = reinterpret_cast<void*>(i.second.second);
				i.second.second=0;
				CUDA_CHECK_ERR(cudaFree(real_adr));
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
	void tocpu(const stream<D>& stream, T* dst_y, const T* from_x, int n);
	template<> 
	void tocpu<DeviceType::CPU, float >(const stream<DeviceType::CPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(float)*n);
		}


	}
	template<> 
	void tocpu<DeviceType::CPU, double >(const stream<DeviceType::CPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(double)*n);
		}

	}
#if ALSM_USE_GPU
	template<> 
	void tocpu<DeviceType::GPU, float >(const stream<DeviceType::GPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(float)*n, cudaMemcpyDeviceToHost));
		}

	}
	template<> 
	void tocpu<DeviceType::GPU, double >(const stream<DeviceType::GPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(double)*n, cudaMemcpyDeviceToHost));
		}
	}
#endif
	template< DeviceType D, typename T> 
	void fromcpu(const stream<D>& stream, T* dst_y, const T* from_x, int n);
	template<> 
	void fromcpu<DeviceType::CPU, float >(const stream<DeviceType::CPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(float)*n);
		}


	}
	template<> 
	void fromcpu<DeviceType::CPU, double >(const stream<DeviceType::CPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(double)*n);
		}

	}
#if ALSM_USE_GPU
	template<> 
	void fromcpu<DeviceType::GPU, float >(const stream<DeviceType::GPU>& stream, float* dst_y, const float* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(float)*n, cudaMemcpyHostToDevice));
		}

	}
	template<> 
	void fromcpu<DeviceType::GPU, double >(const stream<DeviceType::GPU>& stream, double* dst_y, const double* from_x, int n)
	{
		if (dst_y != from_x)
		{
			CUDA_CHECK_ERR(cudaMemcpy(dst_y, from_x, sizeof(double)*n, cudaMemcpyHostToDevice));
		}
	}
#endif
	template< DeviceType D, typename T>
	void to_server(const stream<D>& stream, T* dst_y, const T* from_x, int n,int server_device_index);
	template<>
	void to_server<DeviceType::CPU, float >(const stream<DeviceType::CPU>& stream, float* dst_y, const float* from_x, int n, int server_device_index)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(float)*n);
		}


	}
	template<>
	void to_server<DeviceType::CPU, double >(const stream<DeviceType::CPU>& stream, double* dst_y, const double* from_x, int n, int server_device_index)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(double)*n);
		}

	}
#if ALSM_USE_GPU
	template<>
	void to_server<DeviceType::GPU, float >(const stream<DeviceType::GPU>& stream, float* dst_y, const float* from_x, int n, int server_device_index)
	{
		if (dst_y != from_x)
		{
			if (stream.device_index == server_device_index)
			{
				CUDA_CHECK_ERR(cudaMemcpyAsync(dst_y, from_x, sizeof(float)*n, cudaMemcpyDeviceToDevice, stream.cudastream));
			}
			else
			{
				CUDA_CHECK_ERR(cudaMemcpyPeerAsync(dst_y, server_device_index, from_x, stream.device_index, n*sizeof(float), stream.cudastream));
			}
			
		}


	}
	template<>
	void to_server<DeviceType::GPU, double >(const stream<DeviceType::GPU>& stream, double* dst_y, const double* from_x, int n, int server_device_index)
	{
		if (dst_y != from_x)
		{
			if (stream.device_index == server_device_index)
			{
				CUDA_CHECK_ERR(cudaMemcpyAsync(dst_y, from_x, sizeof(double)*n, cudaMemcpyDeviceToDevice, stream.cudastream));
			}
			else
			{
				CUDA_CHECK_ERR(cudaMemcpyPeerAsync(dst_y, server_device_index, from_x, stream.device_index, n*sizeof(double), stream.cudastream));
			}
		}
	}
#endif
	template< DeviceType D, typename T>
	void from_server(const stream<D>& stream, T* dst_y, const T* from_x, int n, int client_device_index);
	template<>
	void from_server<DeviceType::CPU, float >(const stream<DeviceType::CPU>& stream, float* dst_y, const float* from_x, int n, int client_device_index)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(float)*n);
		}


	}
	template<>
	void from_server<DeviceType::CPU, double >(const stream<DeviceType::CPU>& stream, double* dst_y, const double* from_x, int n, int client_device_index)
	{
		if (dst_y != from_x)
		{
			memcpy(dst_y, from_x, sizeof(double)*n);
		}

	}
#if ALSM_USE_GPU
	template<>
	void from_server<DeviceType::GPU, float >(const stream<DeviceType::GPU>& stream, float* dst_y, const float* from_x, int n, int client_device_index)
	{
		if (dst_y != from_x)
		{
			if (stream.device_index == client_device_index)
			{
				CUDA_CHECK_ERR(cudaMemcpyAsync(dst_y, from_x, sizeof(float)*n, cudaMemcpyDeviceToDevice, stream.cudastream));
			}
			else
			{
				CUDA_CHECK_ERR(cudaMemcpyPeerAsync(dst_y, client_device_index, from_x, stream.device_index, n*sizeof(float), stream.cudastream));
			}

		}


	}
	template<>
	void from_server<DeviceType::GPU, double >(const stream<DeviceType::GPU>& stream, double* dst_y, const double* from_x, int n, int client_device_index)
	{
		if (dst_y != from_x)
		{
			if (stream.device_index == client_device_index)
			{
				CUDA_CHECK_ERR(cudaMemcpyAsync(dst_y, from_x, sizeof(double)*n, cudaMemcpyDeviceToDevice,stream.cudastream));
			}
			else
			{
				CUDA_CHECK_ERR(cudaMemcpyPeerAsync(dst_y, client_device_index, from_x, stream.device_index, n*sizeof(double), stream.cudastream));
			}
		}
	}
#endif

}
#endif
