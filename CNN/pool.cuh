#pragma once

#include "layer.h"
#include "GPU.cuh"

namespace pool
{
    using layer::Layer;

    template<typename T>
    struct PoolLayer : public Layer<T>
    {
        T **I,**dI;
        unsigned *P;
        const unsigned Or,Oc,Och,
                       Ir,Ic,Ich,
                       Pr,Pc,Pch;
        
        PoolLayer(const unsigned Or,const unsigned Oc,const unsigned Och,
                  const unsigned Pr,const unsigned Pc,const unsigned Pch,
                  T **I,T **dI)
            : Layer<T>(),
              Or(Or),Oc(Oc),Och(Och),
              Pr(Pr),Pc(Pc),Pch(Pch),
              Ir(Or*Pr),Ic(Oc*Pc),Ich(Och*Pch),
              I(I),dI(dI),P(nullptr) {}

        virtual ~PoolLayer() {}
    };

    template<typename T>
    struct STCLayer : public PoolLayer<T>
    {
        using PoolLayer::I; using PoolLayer::dI; using PoolLayer::P;
        using PoolLayer::Ir; using PoolLayer::Ic; using PoolLayer::Ich;
        using PoolLayer::Pr; using PoolLayer::Pc; using PoolLayer::Pch;
        using PoolLayer::Or; using PoolLayer::Oc; using PoolLayer::Och;
        using Layer::O; using Layer::dO;

        STCLayer(const unsigned Or,const unsigned Oc,const unsigned Och,
                 const unsigned Pr,const unsigned Pc,const unsigned Pch,
                 T **I,T **dI)
            : PoolLayer<T>(Or,Oc,Och,Pr,Pc,Pch,I,dI)
        {
            O = new T[Or*Oc*Och];
            P = new unsigned[Or*Oc*Och];
        }

        ~STCLayer() {delete[] O,P;}

        void forward() override;
        void backward() override;
    };

    template<typename T>
    struct OMPLayer : public PoolLayer<T>
    {
        using PoolLayer::I; using PoolLayer::dI; using PoolLayer::P;
        using PoolLayer::Ir; using PoolLayer::Ic; using PoolLayer::Ich;
        using PoolLayer::Pr; using PoolLayer::Pc; using PoolLayer::Pch;
        using PoolLayer::Or; using PoolLayer::Oc; using PoolLayer::Och;
        using Layer::O; using Layer::dO;

        OMPLayer(const unsigned Or,const unsigned Oc,const unsigned Och,
                 const unsigned Pr,const unsigned Pc,const unsigned Pch,
                 T **I,T **dI)
            : PoolLayer<T>(Or,Oc,Och,Pr,Pc,Pch,I,dI)
        {
            O = new T[Or*Oc*Och];
            P = new unsigned[Or*Oc*Och];
        }

        ~OMPLayer() {delete[] O,P;}

        void forward() override;
        void backward() override;
    };

    struct CONSTANTS
    {
        const unsigned Ir,Ic,Ich,
                       Or,Oc,Och,
                       Pr,Pc,Pch;
    };
    template<typename T>
    struct GPULayer : public PoolLayer<T>
    {
        using PoolLayer::I; using PoolLayer::dI; using PoolLayer::P;
        using PoolLayer::Ir; using PoolLayer::Ic; using PoolLayer::Ich;
        using PoolLayer::Pr; using PoolLayer::Pc; using PoolLayer::Pch;
        using PoolLayer::Or; using PoolLayer::Oc; using PoolLayer::Och;
        using Layer::O; using Layer::dO;

        T *d_O,**d_I;
        cudaStream_t stream;

        GPULayer(const unsigned Or,const unsigned Oc,const unsigned Och,
                 const unsigned Pr,const unsigned Pc,const unsigned Pch,
                 T **I,T **dI,cudaStream_t stream,T **d_I) // 'I' kept for compatibility w/ other layers.
            : PoolLayer<T>(Or,Oc,Och,Pr,Pc,Pch,I,dI),
              stream(stream),d_I(d_I),d_O(nullptr)
        {
            GPU::allocHostPinned(&O,Or*Oc*Och);
            GPU::allocHostPinned(&P,Or*Oc*Och);
        }

        ~GPULayer()
        {
            GPU::destroyHostPinned(P);
            GPU::destroyHostPinned(O);
        }

        /*
        Preconditions:
            - 'I' already in device memory
        Effects:
            - 'I' removed from device memory
            - 'O' allocated in device memory
        */
        void forward() override;
        /*
        Preconditions:
            - 'dO' allocated in device memory
        Effects:
            - 'dO' removed from device memory
            - 'dI' allocated in device memory
        */
        void backward() override;
    };
}

#include "intellisense_fix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <omp.h>

template<typename T>
void pool::STCLayer<T>::forward()
{
    for(unsigned or = 0;or < Or;++or)
    {
        const unsigned _ir = Ic*Ich*Pr*or,
            _or = Oc*Och*or;
        for(unsigned oc = 0;oc < Oc;++oc)
        {
            const unsigned _ic = _ir+Ich*Pc*oc,
                _oc = _or+Och*oc;
            for(unsigned och = 0;och < Och;++och)
            {
                const unsigned _ich = _ic+Pch*och,
                    _och = _oc+och;
                T max = (*I)[_ich];
                unsigned midx = 0;
                for(unsigned pr = 0;pr < Pr;++pr)
                {
                    const unsigned ir = _ich+Ic*Ich*pr;
                    for(unsigned pc = 0;pc < Pc;++pc)
                    {
                        const unsigned ic = ir+Ich*pc;
                        for(unsigned pch = 0;pch < Pch;++pch)
                        {
                            const unsigned ich = ic+pch;
                            const T &i = (*I)[ich];
                            if(max < i)
                            {
                                max = i;
                                midx = Pc*Pch*pr+Pch*pc+pch;
                            }
                        }
                    }
                }
                O[_och] = max;
                P[_och] = midx;
            }
        }
    }
}

template<typename T>
void pool::STCLayer<T>::backward()
{
    *dI = new T[Ir*Ic*Ich]();
    for(unsigned or = 0;or < Or;++or)
    {
        const unsigned _ir = Ic*Ich*Pr*or,
            _or = Oc*Och*or;
        for(unsigned oc = 0;oc < Oc;++oc)
        {
            const unsigned _ic = _ir+Ich*Pc*oc,
                _oc = _or+Och*oc;
            for(unsigned och = 0;och < Och;++och)
            {
                const unsigned _ich = _ic+Pch*och,
                    _och = _oc+och,

                    p = P[_och],
                    pr = p/Pch/Pc,
                    pc = (p/Pch)%Pc,
                    pch = p%Pch,

                    ix = _ich+Ic*Ich*pr+
                    Ich*pc+
                    pch;
                (*dI)[ix] = dO[_och];
            }
        }
    }
    delete[] dO;
    dO = nullptr;
}

template<typename T>
void pool::OMPLayer<T>::forward()
{
    #pragma omp parallel for
    for(long long o = 0;o < (long long)Or*Oc*Och;++o)
    {
        const unsigned or = (unsigned)(o/Och/Oc),
            oc = (unsigned)((o/Och)%Oc),
            och = (unsigned)(o%Och),

            _ir = Ic*Ich*Pr*or,
            _or = Oc*Och*or,
            _ic = _ir+Ich*Pc*oc,
            _oc = _or+Och*oc,
            _ich = _ic+Pch*och,
            _och = _oc+och;

        T max = (*I)[_ich];
        unsigned midx = 0;
        for(unsigned pr = 0;pr < Pr;++pr)
        {
            const unsigned ir = _ich+Ic*Ich*pr;
            for(unsigned pc = 0;pc < Pc;++pc)
            {
                const unsigned ic = ir+Ich*pc;
                for(unsigned pch = 0;pch < Pch;++pch)
                {
                    const unsigned ich = ic+pch;
                    const T &i = (*I)[ich];
                    if(max < i)
                    {
                        max = i;
                        midx = Pc*Pch*pr+Pch*pc+pch;
                    }
                }
            }
        }
        O[_och] = max;
        P[_och] = midx;
    }
}

template<typename T>
void pool::OMPLayer<T>::backward()
{
    *dI = new T[Ir*Ic*Ich]();
    #pragma omp parallel for
    for(long long o = 0;o < (long long)Or*Oc*Och;++o)
    {
        const unsigned or = (unsigned)(o/Och/Oc),
            oc = (unsigned)((o/Och)%Oc),
            och = (unsigned)(o%Och),

            p = P[o],
            pr = p/Pch/Pc,
            pc = (p/Pch)%Pc,
            pch = p%Pch,

            i = Ic*Ich*(Pr*or+pr)+
            Ich*(Pc*oc+pc)+
            Pch*och+pch;
        (*dI)[i] = dO[o];
    }
    delete[] dO;
    dO = nullptr;
}

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
    //if(pr < C.Pr && pc < C.Pc && pch < C.Pch)
    if(IN_BOUNDS(x) && IN_BOUNDS(y) && IN_BOUNDS(z))
        v = pair<T>{I[i],p};
    v = blockReduceMax(v);
    if(!(pr|pc|pch))
    {
        O[_o] = v.v;
        P[_o] = v.k;
    }
}

template<typename T>
void pool::GPULayer<T>::forward()
{
    // Allocate 'P'&'O'
    unsigned *d_P;
    GPU::allocDeviceMem(&d_P,Or*Oc*Och,stream);
    GPU::allocDeviceMem(&d_O,Or*Oc*Och,stream);

    // Start kernel
    dim3 block(Pc,Pr,Pch);
    dim3 grid(Oc,Or,Och);
    const CONSTANTS C{Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch};
    fwdKernel CONFIG4(
        grid,block,
        REDUCE_SM(Pr*Pc*Pch,pair<T>),
        stream
    )(*d_I,d_O,d_P,C);

    // Cleanup 'I'
    GPU::destroyDeviceMem(*d_I,stream);
    *d_I = nullptr;

    // Transfer and cleanup 'P'
    GPU::destroyTransfer(d_P,P,Or*Oc*Och,stream);

    // Transfer 'O'
    GPU::transfer<T,cudaMemcpyDeviceToHost>(d_O,O,Or*Oc*Och,stream);
}

template<typename T>
__global__ void bkwdKernel(T *__restrict__ dI,
                           const T *__restrict__ dO,
                           const unsigned *__restrict__ P,
                           const pool::CONSTANTS C)
{
    const unsigned o = blockDim.x*blockIdx.x+threadIdx.x;
    if(o < C.Or*C.Oc*C.Och)
    {
        const unsigned och = o%C.Och,
                        oc = (o/C.Och)%C.Oc,
                        or = o/C.Och/C.Oc,

                         p = P[o],
                        pr = p/C.Pch/C.Pc,
                        pc = (p/C.Pch)%C.Pc,
                       pch = p%C.Pch,

                         i = C.Ic*C.Ich*(C.Pr*or+pr)+
                                  C.Ich*(C.Pc*oc+pc)+
                                        C.Pch*och+pch;
        dI[i] = dO[o];
    }
}

template<typename T>
void pool::GPULayer<T>::backward()
{
    // Allocate 'dI'
    GPU::callocDeviceMem(dI,Ir*Ic*Ich,stream);

    // Allocate and transfer 'P'
    unsigned *d_P;
    GPU::allocTransfer(P,&d_P,Or*Oc*Och,stream);

    // Start kernel
    dim3 block(std::min<unsigned>((unsigned)GPU::properties.maxThreadsPerBlock,Or*Oc*Och));
    dim3 grid(std::max<unsigned>(1,(Or*Oc*Och-1+block.x)/block.x));
    const CONSTANTS C{Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch};
    bkwdKernel CONFIG4(
        grid,block,
        0,stream
    )(*dI,dO,d_P,C);

    // Cleanup 'P'&'dO'
    GPU::destroyDeviceMem(d_P,stream);
    GPU::destroyDeviceMem(dO,stream);
    dO = nullptr;
}