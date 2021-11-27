#pragma once

#include "layer.h"
#include "GPU.cuh"

namespace fcl
{
    using layer::Layer;

    template<typename T>
    struct FCLLayer : public Layer<T>
    {
        T **I,*W,*B,**dI,*IB;
        const unsigned Is,Os;
        const T LR;

        FCLLayer(const unsigned Is,const unsigned Os,T **I,T **dI,const T LR)
            : Layer<T>(),
              I(I),dI(dI),
              W(nullptr),B(nullptr),IB(nullptr),
              Is(Is),Os(Os),LR(LR) {}

        virtual ~FCLLayer() {}

        void init();
    };

    template<typename T>
    struct STCLayer : public FCLLayer<T>
    {
        using FCLLayer::I; using FCLLayer::W; using FCLLayer::B;
        using FCLLayer::dI; using FCLLayer::IB;
        using FCLLayer::Is; using FCLLayer::Os;
        using Layer::O; using Layer::dO; using FCLLayer::LR;

        STCLayer(const unsigned Is,const unsigned Os,T **I,T **dI,const T LR)
            : FCLLayer<T>(Is,Os,I,dI,LR)
        {
            W = new T[Is*Os];
            B = new T[Is];
            O = new T[Os];
            IB = new T[Is];
        }
        ~STCLayer() {delete[] W,B,O,IB;}

        void forward() override;
        void backward() override;
    };

    template<typename T>
    struct OMPLayer : public FCLLayer<T> {
        using FCLLayer::I; using FCLLayer::W; using FCLLayer::B;
        using FCLLayer::dI; using FCLLayer::IB;
        using FCLLayer::Is; using FCLLayer::Os;
        using Layer::O; using Layer::dO; using FCLLayer::LR;

        OMPLayer(const unsigned Is,const unsigned Os,T **I,T **dI,const T LR)
            : FCLLayer<T>(Is,Os,I,dI,LR)
        {
            W = new T[Is*Os];
            B = new T[Is];
            O = new T[Os];
            IB = new T[Is];
        }
        ~OMPLayer() {delete[] W,B,O,IB;}

        void forward() override;
        void backward() override;
    };

    template<typename T>
    struct GPULayer : public FCLLayer<T> {
        using FCLLayer::I; using FCLLayer::W; using FCLLayer::B;
        using FCLLayer::dI; using FCLLayer::IB;
        using FCLLayer::Is; using FCLLayer::Os;
        using Layer::O; using Layer::dO; using FCLLayer::LR;

        T **d_I,*d_O;
        cudaStream_t stream;

        GPULayer(const unsigned Is,const unsigned Os,T **I,T **dI,const T LR,
                 T **d_I,cudaStream_t stream)
            : FCLLayer<T>(Is,Os,I,dI,LR),d_I(d_I),d_O(nullptr),
              stream(stream)
        {
            GPU::allocHostPinned(&W,Is*Os);
            GPU::allocHostPinned(&B,Is);
            GPU::allocHostPinned(&O,Os);
            GPU::allocHostPinned(&IB,Is);
        }
        ~GPULayer()
        {
            GPU::destroyHostPinned(W);
            GPU::destroyHostPinned(B);
            GPU::destroyHostPinned(O);
            GPU::destroyHostPinned(IB);
        }

        /*
        Preconditions:
            - 'I' already in device memory
        Effects:
            - 'I' removed from device memory
            - 'O' allocated to device memory
        */
        void forward() override;
        /*
        Preconditions:
            - 'dO' already in device memory
        Effects:
            - 'dO' deallocated
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
#include <random>

template<typename T>
void fcl::FCLLayer<T>::init()
{
    static std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,std::sqrt(2./Is));
    for(unsigned w = 0;w < Is*Os;++w)
        W[w] = distribution(generator);
}

template<typename T>
void fcl::STCLayer<T>::forward()
{
    for(unsigned i = 0;i < Is;++i)
        IB[i] = std::max<T>(T(0),(*I)[i]+B[i]);
    for(unsigned o = 0;o < Os;++o)
    {
        T v(0);
        const unsigned wo = Is*o;
        for(unsigned i = 0;i < Is;++i)
            v += IB[i] * W[wo+i];
        O[o] = v;
    }
}

template<typename T>
void fcl::STCLayer<T>::backward()
{
    for(unsigned o = 0;o < Os;++o)
        dO[o] *= LR;

    *dI = new T[Is];
    for(unsigned i = 0;i < Is;++i)
    {
        const T &Iv = IB[i];
        T dIv(0);
        if(Iv != T(0))
        {
            for(unsigned o = 0;o < Os;++o)
            {
                const unsigned w = Is*o+i;
                const T &dLdO = dO[o];
                dIv += dLdO * W[w];
                W[w] -= dLdO * Iv;
            }
            B[i] -= dIv;
        }
        (*dI)[i] = dIv;
    }

    delete[] dO;
    dO = nullptr;
}

template<typename T>
void fcl::OMPLayer<T>::forward()
{
    #pragma omp parallel for
    for(long i = 0;i < (long)Is;++i)
        IB[i] = std::max<T>(T(0),(*I)[i]+B[i]);
    #pragma omp parallel for
    for(long o = 0;o < (long)Os;++o)
    {
        T v(0);
        const unsigned wo = Is*(unsigned)o;
        for(unsigned i = 0;i < Is;++i)
            v += IB[i] * W[wo+i];
        O[o] = v;
    }
}

template<typename T>
void fcl::OMPLayer<T>::backward()
{
    #pragma omp parallel for
    for(long o = 0;o < (long)Os;++o)
        dO[o] *= LR;

    *dI = new T[Is];
    #pragma omp parallel for
    for(long i = 0;i < (long)Is;++i)
    {
        const T &Iv = IB[i];
        T dIv(0);
        if(Iv != T(0))
        {
            for(unsigned o = 0;o < Os;++o)
            {
                const unsigned w = Is*o+(unsigned)i;
                const T dLdO = dO[o];
                dIv += dLdO * W[w];
                W[w] -= dLdO * Iv;
            }
            B[i] -= dIv;
        }
        (*dI)[i] = dIv;
    }

    delete[] dO;
    dO = nullptr;
}

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
    //if(ix < _Is && ox < _Os)
    if(IN_BOUNDS(x))
        v = i[ix] * w[_Is*ox+ix];
    v = blockReduceSum(v);
    if(!ix) o[ox] = v;
}

template<typename T>
void fcl::GPULayer<T>::forward()
{
    // Allocate and transfer bias and temp vectors
    T *d_B,*d_IB;
    GPU::allocTransfer(B,&d_B,Is,stream);
    GPU::allocTransfer(IB,&d_IB,Is,stream);

    // Compute ReLU and bias
    {
        const unsigned maxBlock = GPU::properties.maxThreadsPerBlock;
        dim3 block(std::min<unsigned>(maxBlock,Is));
        dim3 grid((block.x-1+maxBlock)/maxBlock);

        fwd_I CONFIG4(grid,block,0,stream)(*d_I,d_IB,d_B,Is);
    }

    // Cleanup input and bias vectors
    GPU::destroyDeviceMem(d_B,stream);
    GPU::destroyDeviceMem(*d_I,stream);
    *d_I = nullptr;

    // Allocate and transfer weight matrix
    T *d_W;
    GPU::allocTransfer(W,&d_W,Is*Os,stream);

    // Allocate ouput vector
    GPU::allocDeviceMem(&d_O,Os,stream);

    // Compute 'I*W'
    {
        dim3 block(Is);
        dim3 grid(Os);

        fwd_main CONFIG4(
            grid,block,
            REDUCE_SM(Is,T),
            stream
        )(d_IB,d_W,d_O,Is,Os);
    }

    // Cleanup weights
    GPU::destroyDeviceMem(d_W,stream);

    // Transfer and clean up input
    GPU::destroyTransfer(d_IB,IB,Is,stream);

    // Transfer output
    GPU::transfer<T,cudaMemcpyDeviceToHost>(d_O,O,Os,stream);
}

template<typename T>
__global__ void bkwd_main(const T *__restrict__ dldo,
                          const T *__restrict__ i,
                          T *__restrict__ w,
                          T *__restrict__ b,
                          T *__restrict__ di,
                          const unsigned _Is,
                          const unsigned _Os,
                          const T LR)
{
    const unsigned ix = blockIdx.x,
                   ox = threadIdx.x,
                   wx = _Is*ox+ix;
    T dIv(0);
    //if(ox < _Os && i[ix])
    if(IN_BOUNDS(x) && i[ix])
    {
        dIv = dldo[ox] * w[wx];
        w[wx] -= dldo[ox] * i[ix] * LR;
    }
    dIv = blockReduceSum(dIv);
    if(!ox) b[ix] -= (di[ix] = dIv) * LR;
}

template<typename T>
void fcl::GPULayer<T>::backward()
{
    // Allocate and transfer
    T *d_IB,*d_B,*d_W;
    GPU::allocTransfer(IB,&d_IB,Is,stream);
    GPU::allocTransfer(B,&d_B,Is,stream);
    GPU::allocTransfer(W,&d_W,Is*Os,stream);

    // Allocate dL/dI
    GPU::allocDeviceMem(dI,Is,stream);

    // Compute dL/dI, update weights, and update bias
    {
        dim3 block(Os);
        dim3 grid(Is);

        bkwd_main CONFIG4(
            grid,block,
            REDUCE_SM(Os,T),
            stream
        )(dO,d_IB,d_W,d_B,*dI,Is,Os,LR);
    }

    // Cleanup dL/dO
    GPU::destroyDeviceMem(dO,stream);
    dO = nullptr;

    // Transfer and cleanup weights
    GPU::destroyTransfer(d_W,W,Is*Os,stream);

    // Transfer and cleanup bias
    GPU::destroyTransfer(d_B,B,Is,stream);

    // Cleanup input
    GPU::destroyDeviceMem(d_IB,stream);
}