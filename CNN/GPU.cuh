#pragma once

#ifdef DEBUG
#include "debug.h"
#endif // DEBUG

#include <cuda_runtime.h>

template<typename T>
__inline__ __device__ T warpReduceSum(T v)
{
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ <= 860
    // If the architecture is known, use a compile-time
    // constant instead of warpSize so that the loop may
    // be unrolled.
    #pragma unroll
    for(unsigned t = 32;t >>= 1;)
    #else
    for(unsigned t = warpSize;t >>= 1;)
    #endif // __CUDA_ARCH__
        v += __shfl_down_sync(~0,v,t);
    return v;
}
template<typename T>
__inline__ __device__ T blockReduceSum(T v)
{
    extern __shared__ T res[];
    const unsigned Bx = blockDim.x,By = blockDim.y,Bz = blockDim.z,
                   bx = threadIdx.x,by = threadIdx.y,bz = threadIdx.z,
                    M = (Bx*By*Bz - 1 + warpSize) / warpSize,
                    t = By*Bz*bx+Bz*by+bz;
    v = warpReduceSum<T>(v);
    if(!(t % warpSize)) res[t/warpSize] = v;
    __syncthreads();
    v = t < M? res[t] : T(0);
    __syncthreads();
    return warpReduceSum<T>(v);
}

template<typename T> struct pair {T v; unsigned k;};
template<typename T>
__inline__ __device__ pair<T> warpReduceMax(pair<T> v)
{
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ <= 860
    // If the architecture is known, use a compile-time
    // constant instead of warpSize so that the loop may
    // be unrolled.
    #pragma unroll
    for(unsigned t = 32;t >>= 1;)
    #else
    for(unsigned t = warpSize;t >>= 1;)
    #endif // __CUDA_ARCH__
    {
        const T uv = __shfl_down_sync(~0,v.v,t);
        const unsigned uk = __shfl_down_sync(~0,v.k,t);
        if(v.v < uv) {v.k = uk;v.v = uv;}
    }
    return v;
}
template<typename T>
__inline__ __device__ pair<T> blockReduceMax(pair<T> v)
{
    extern __shared__ pair<T> res[];
    const unsigned Bx = blockDim.x, By = blockDim.y, Bz = blockDim.z,
                   bx = threadIdx.x,by = threadIdx.y,bz = threadIdx.z,
                    M = (Bx*By*Bz - 1 + warpSize) / warpSize,
                    t = By*Bz*bx+Bz*by+bz;
    v = warpReduceMax(v);
    if(!(t % warpSize)) res[t/warpSize] = v;
    __syncthreads();
    v = t < M? res[t] : res[M-1];
    __syncthreads();
    return warpReduceMax(v);
}

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
            err(cudaGetErrorName(e),':',cudaGetErrorString(e));
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

    /* Computes the required size of shared memory for the block reduce. */
    template<typename T>
    inline unsigned reduceSM(unsigned vol,unsigned warpsize)
    {
        return (vol*sizeof(T)-1U+warpsize)/warpsize;
    }
}