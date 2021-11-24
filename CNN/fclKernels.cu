#include "intellisense_fix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "fcl.cuh"

template<typename T>
__global__ void fwd_I(const T *__restrict__ i,
                      T *__restrict__ ib,
                      const T *__restrict__ b,
                      const unsigned _Is)
{
    const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < _Is)
    {
        const T iv = i[idx] + b[idx];
        ib[idx] = iv > T(0)? iv : T(0);
    }
}
template<typename T>
__global__ void fwd_main(const T *__restrict__ i,
                         const T *__restrict__ w,
                         T *__restrict__ o,
                         const unsigned _Is,
                         const unsigned _Os)
{
    const unsigned ix = threadIdx.x,ox = blockIdx.x;
    T v(0);
    if(ix < _Is && ox < _Os)
        v = i[ix] * w[_Is*ox+ix];
    v = blockReduceSum<T>(v);
    if(!ix) o[ox] = v;
}

template<typename T>
void fcl::fwdImpl(fcl::GPULayer<T> *l)
{
    // Allocate and transfer bias and temp vectors
    T *d_B,*d_IB;
    GPU::allocTransfer(l->B,&d_B,l->Is,l->stream);
    GPU::allocTransfer(l->IB,&d_IB,l->Is,l->stream);

    // Compute ReLU and bias
    {
        const unsigned maxBlock = GPU::properties.maxThreadsPerBlock;
        dim3 block(std::min<unsigned>(maxBlock,l->Is));
        dim3 grid((block.x-1+maxBlock)/maxBlock);

        fwd_I CONFIG4(grid,block,0,l->stream)(*l->d_I,d_IB,d_B,l->Is);
    }

    // Cleanup input and bias vectors
    GPU::destroyDeviceMem(d_B,l->stream);
    GPU::destroyDeviceMem(*l->d_I,l->stream);

    // Allocate and transfer weight matrix
    T *d_W;
    GPU::allocTransfer(l->W,&d_W,l->Is*l->Os,l->stream);

    // Allocate ouput vector
    GPU::allocDeviceMem(&l->d_O,l->Os,l->stream);

    // Compute 'I*l->W'
    {
        dim3 block(l->Is);
        dim3 grid(l->Os);

        fwd_main CONFIG4(
            grid,block,
            GPU::reduceSM<T>(l->Is,(unsigned)GPU::properties.warpSize),
            l->stream
        )(d_IB,d_W,l->d_O,l->Is,l->Os);
    }

    // Cleanup weights
    GPU::destroyDeviceMem(d_W,l->stream);

    // Transfer and clean up input
    GPU::destroyTransfer(&d_IB,l->IB,l->Is,l->stream);

    // Transfer output
    GPU::transfer<T,cudaMemcpyDeviceToHost>(l->d_O,l->O,l->Os,l->stream);
}

template<typename T>
__global__ void bkwd_dO(T *__restrict__ dldo,
                        const T lr,
                        const unsigned _Os)
{
    const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < _Os) dldo[idx] *= lr;
}
template<typename T>
__global__ void bkwd_main(const T *__restrict__ dldo,
                          const T *__restrict__ i,
                          T *__restrict__ w,
                          T *__restrict__ b,
                          T *__restrict__ di,
                          const unsigned _Is,
                          const unsigned _Os)
{
    const unsigned ix = blockIdx.x,
        ox = threadIdx.x,
        wx = _Is*ox+ix;
    T dIv(0);
    if(ox < _Os && i[ix] != T(0))
    {
        const T dLdO = dldo[ox];
        dIv = dLdO * w[wx];
        w[wx] -= dLdO * i[ix];
    }
    dIv = blockReduceSum(dIv);
    if(!ox) b[ix] -= di[ix] = dIv;
}

template<typename T>
void fcl::bkwdImpl(fcl::GPULayer<T> *l)
{
    // Scale by learning rate
    {
        const unsigned maxBlock = GPU::properties.maxThreadsPerBlock;
        dim3 block(std::min<unsigned>(maxBlock,l->Os));
        dim3 grid((block.x-1+maxBlock)/maxBlock);

        bkwd_dO CONFIG4(grid,block,0,l->stream)(l->dO,l->LR,l->Os);
    }

    // Allocate and transfer
    T *d_IB,*d_B,*d_W;
    GPU::allocTransfer(l->IB,&d_IB,l->Is,l->stream);
    GPU::allocTransfer(l->B,&d_B,l->Is,l->stream);
    GPU::allocTransfer(l->W,&d_W,l->Is*l->Os,l->stream);

    // Allocate dL/dI
    GPU::allocDeviceMem(l->dI,l->Is,l->stream);

    // Compute dL/dI, update weights, and update bias
    {
        dim3 block(l->Os);
        dim3 grid(l->Is);

        bkwd_main CONFIG4(
            grid,block,
            GPU::reduceSM<T>(l->Os,(unsigned)GPU::properties.warpSize),
            l->stream
        )(l->dO,d_IB,d_W,d_B,*l->dI,l->Is,l->Os);
    }

    // Cleanup dL/dO
    GPU::destroyDeviceMem(l->dO,l->stream);

    // Transfer and cleanup weights
    GPU::destroyTransfer(&d_W,l->W,l->Is*l->Os,l->stream);

    // Transfer and cleanup bias
    GPU::destroyTransfer(&d_B,l->B,l->Is,l->stream);

    // Cleanup input
    GPU::destroyDeviceMem(d_IB,l->stream);
}