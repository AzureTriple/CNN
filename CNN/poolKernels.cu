#include "intellisense_fix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "pool.cuh"

template<typename T>
__global__ void fwdKernel(const T *__restrict__ I,
                          T *__restrict__ O,
                          unsigned *__restrict__ P,
                          const pool::CONSTANTS C)
{
    const unsigned och = blockIdx.z,
                    or = blockIdx.y,
                    oc = blockIdx.x,

                    _i = C.Ic*C.Ich*C.Pr*or+
                              C.Ich*C.Pc*oc+
                                    C.Pch*och,
                    _o = C.Oc*C.Och*or+
                              C.Och*oc+
                                    och,

                   pch = threadIdx.z,
                    pr = threadIdx.y,
                    pc = threadIdx.x,

                     i = _i+C.Ic*C.Ich*pr+
                                 C.Ich*pc+
                                       pch,
                    p = C.Pc*C.Pch*pr+
                             C.Pch*pc+
                                   pch;
    pair<T> v{I[_i],0};
    if(pr < C.Pr && pc < C.Pc && pch < C.Pch)
        v = pair<T>{I[i],p};
    v = blockReduceMax(v);
    if(!(pr|pc|pch))
    {
        O[_o] = v.v;
        P[_o] = v.k;
    }
}

template<typename T>
void pool::fwdImpl(pool::GPULayer<T> *l)
{
    // Allocate 'P'&'O'
    unsigned *d_P;
    GPU::allocDeviceMem(&d_P,l->Or*l->Oc*l->Och,l->stream);
    GPU::allocDeviceMem(&l->d_O,l->Or*l->Oc*l->Och,l->stream);

    // Start kernel
    dim3 block(l->Pc,l->Pr,l->Pch);
    dim3 grid(l->Oc,l->Or,l->Och);
    const pool::CONSTANTS C{l->Ir,l->Ic,l->Ich,l->Or,l->Oc,l->Och,l->Pr,l->Pc,l->Pch};
    fwdKernel CONFIG4(
        grid,block,
        GPU::reduceSM<pair<T>>(l->Pr*l->Pc*l->Pch,(unsigned)GPU::properties.warpSize),
        l->stream
    )(*l->d_I,l->d_O,d_P,C);

    // Cleanup 'I'
    GPU::destroyDeviceMem(*l->d_I,l->stream);

    // Transfer and cleanup 'P'
    GPU::destroyTransfer(&d_P,l->P,l->Or*l->Oc*l->Och,l->stream);

    // Transfer 'O'
    GPU::transfer<T,cudaMemcpyDeviceToHost>(l->d_O,l->O,l->Or*l->Oc*l->Och,l->stream);
}

template<typename T>
__global__ void bkwdKernel(T *__restrict__ dI,
                           const T *__restrict__ dO,
                           const unsigned *__restrict__ P,
                           const pool::CONSTANTS C)
{
    const unsigned o = blockDim.x*blockIdx.x+threadIdx.x,

                 och = o%C.Och,
                  oc = (o/C.Och)%C.Oc,
                  or = o/C.Och/C.Oc,

                   p = P[o],
                  pr = p/C.Pch/C.Pc,
                  pc = (p/C.Pch)%C.Pc,
                 pch = p%C.Pch,

                   i = C.Ic*C.Ich*(C.Pr*or+pr)+
                            C.Ich*(C.Pc*oc+pc)+
                                  C.Pch*och+pch;
    if(threadIdx.x < blockDim.x)
        dI[i] = dO[o];
}

template<typename T>
void pool::bkwdImpl(pool::GPULayer<T> *l)
{
    // Allocate 'l->dI'
    GPU::callocDeviceMem(l->dI,l->Ir*l->Ic*l->Ich,l->stream);

    // Allocate and transfer 'P'
    unsigned *d_P;
    GPU::allocTransfer(l->P,&d_P,l->Or*l->Oc*l->Och,l->stream);

    // Start kernel
    dim3 block(std::min<unsigned>((unsigned)GPU::properties.maxThreadsPerBlock,l->Or*l->Oc*l->Och));
    dim3 grid(std::max<unsigned>(1,(l->Or*l->Oc*l->Och-1+block.x)/block.x));
    const pool::CONSTANTS C{l->Ir,l->Ic,l->Ich,l->Or,l->Oc,l->Och,l->Pr,l->Pc,l->Pch};
    bkwdKernel CONFIG4(
        grid,block,
        0,l->stream
    )(*l->dI,l->dO,d_P,C);

    // Cleanup 'P'&'l->dO'
    GPU::destroyDeviceMem(d_P,l->stream);
    GPU::destroyDeviceMem(l->dO,l->stream);
}