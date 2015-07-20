#ifndef __H_CACHELINE_H__
#define __H_CACHELINE_H__
#define CACHE_SIZE 128
namespace alsm
{
	template <typename T> class cache_align_storage
	{
	public:
		T data;
		char padding[sizeof(T) < CACHE_SIZE ? CACHE_SIZE - sizeof(T) : 1];
		cache_align_storage() :data()
		{

		}
		cache_align_storage(const cache_align_storage& in_ca_storage) = delete;
		cache_align_storage(cache_align_storage&& in_ca_storage) = delete;
		cache_align_storage& operator=(const cache_align_storage& in_ca_storage) = delete;
		cache_align_storage& operator=(const cache_align_storage&& in_ca_storage) = delete;
	};
}

#endif