#pragma once

#ifdef DEBUG
#include "debug.h"
#endif // DEBUG

#include <cuda_runtime.h>

namespace GPU
{
    static inline void check(cudaError_t e)
    {
        #ifdef DEBUG
        if(e != cudaSuccess)
            err(cudaGetErrorString(e));
        #endif // DEBUG
    }
    static cudaDeviceProp getProperties()
    {
        cudaDeviceProp p;
        int d;
        check(cudaGetDevice(&d));
        check(cudaGetDeviceProperties(&p,d));
        return p;
    }
    const cudaDeviceProp properties = getProperties();

    cudaStream_t createStream();
    void destroyStream(cudaStream_t stream);
    
    template<typename T>
    void allocHostPinned(T **arr,size_t count,unsigned mode = cudaHostAllocDefault)
    {
        check(cudaHostAlloc((void**)arr,count*sizeof(T),mode));
    }
    void destroyHostPinned(void *arr);

    template<typename T>
    void allocDeviceMem(T **arr,size_t count,cudaStream_t stream)
    {
        check(cudaMallocAsync((void**)arr,count*sizeof(T),stream));
    }
    void destroyDeviceMem(void *arr,cudaStream_t stream);
    
    template<typename T,cudaMemcpyKind MODE = cudaMemcpyDefault>
    void transfer(T *src,T *dst,size_t count,cudaStream_t stream)
    {
        GPU::check(cudaMemcpyAsync((void*)dst,(void*)src,count*sizeof(T),MODE,stream));
    }

    void sync();
    void sync(cudaStream_t stream);
}