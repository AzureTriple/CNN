#pragma once

#ifdef DEBUG
#include "debug.h"
#endif // DEBUG

#include <cuda_runtime.h>

namespace GPU
{
    /*
    When the DEBUG macro is defined,this function will check and report CUDA 
    runtime errors.
    */
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

    /* Creates a CUDA stream. */
    cudaStream_t createStream();
    /* Destroys a CUDA stream. */
    void destroyStream(cudaStream_t stream);
    
    /*
    Allocates pinned memory on the host. Pinned memory is page-locked, which
    means that transfers between the host and device are significantly faster.

    Adding too much pinned memory can degrade system performance, so use this
    sparingly.
    */
    template<typename T>
    void allocHostPinned(T **arr,size_t count,unsigned mode = cudaHostAllocDefault)
    {
        check(cudaHostAlloc((void**)arr,count*sizeof(T),mode));
    }
    /* Deallocates pinned memory. */
    void destroyHostPinned(void *arr);

    /* Allocates memory on the device. */
    template<typename T>
    void allocDeviceMem(T **arr,size_t count,cudaStream_t stream)
    {
        check(cudaMallocAsync((void**)arr,count*sizeof(T),stream));
    }
    /* Deallocates memory on the device. */
    void destroyDeviceMem(void *arr,cudaStream_t stream);
    
    /* Transfers data. */
    template<typename T,cudaMemcpyKind MODE = cudaMemcpyDefault>
    void transfer(T *src,T *dst,size_t count,cudaStream_t stream)
    {
        GPU::check(cudaMemcpyAsync((void*)dst,(void*)src,count*sizeof(T),MODE,stream));
    }

    /* Synchronizes the device with the host. */
    void sync();
    /* Synchronizes events on the specified stream. */
    void sync(cudaStream_t stream);
}