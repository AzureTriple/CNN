#pragma once

#include "intellisense_fix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <omp.h>
#include "GPU.cuh"

namespace fcl
{
    /*
    Note:
        The weight matrix is indexed <output,input> in decreasing order
    */
    template<typename T>
    struct Layer
    {
        T *I,*W,*B,**O,*dI,**dO,LR;
        unsigned Is,Os;

        Layer(unsigned Is,unsigned Os,T **O,T **dO,T LR)
            : I(nullptr),W(nullptr),B(nullptr),O(O),
              Is(Is),Os(Os),dI(nullptr),dO(dO),LR(LR) {}

        virtual ~Layer() {}

        virtual void forward() = 0;
        virtual void backward() = 0;
    };

    template<typename T>
    struct CPULayer : public Layer<T>
    {
        using Layer::I; using Layer::W; using Layer::B; using Layer::O;
        using Layer::dI; using Layer::dO;
        using Layer::Is; using Layer::Os;

        CPULayer(unsigned Is,unsigned Os,T **O,T **dO,T LR)
            : Layer<T>(Is,Os,O,dO,LR)
        {
            alloc_I();
            alloc_W();
            alloc_B();
        }

        virtual ~CPULayer()
        {
            dealloc_I();
            dealloc_W();
            dealloc_B();
        }

        void alloc_I() {I = (T*)malloc(sizeof(T)*Is);}
        void alloc_W() {
            W = (T*)malloc(sizeof(T)*Is*Os);
            W[0] = T(0);
        }
        void alloc_B() {B = (T*)malloc(sizeof(T)*Is);}

        void dealloc_I() {free((void*)I);}
        void dealloc_W() {free((void*)W);}
        void dealloc_B() {free((void*)B);}

        void alloc_dI() {dI = (T*)malloc(sizeof(T)*Is);}
        // DEBUG: potentially unsafe
        void dealloc_dO() {free((void*)*dO);}
    };

    template<typename T>
    struct STCLayer : public CPULayer<T>
    {
        using Layer::I; using Layer::W; using Layer::B; using Layer::O;
        using Layer::dI; using Layer::dO; using Layer::LR;
        using Layer::Is; using Layer::Os;

        using CPULayer::CPULayer;
        using CPULayer::alloc_dI;
        using CPULayer::dealloc_dO;

        void forward() override
        {
            for(unsigned i = 0;i < Is;++i)
                I[i] = std::max<T>(T(0),I[i]+B[i]);
            for(unsigned o = 0;o < Os;++o)
            {
                T v(0);
                const unsigned wo = Is*o;
                for(unsigned i = 0;i < Is;++i)
                    v += I[i] * W[wo+i];
                (*O)[o] = v;
            }
        }
        /*
        Preconditions:
            - 'dO' already allocated
        Effects:
            - 'dO' deallocated
            - 'dI' allocated
        */
        void backward() override
        {
            alloc_dI();

            for(unsigned o = 0;o < Os;++o)
                (*dO)[o] *= LR;
            for(unsigned i = 0;i < Is;++i)
            {
                const T &Iv = I[i];
                T dIv(0);
                if(Iv != T(0))
                {
                    for(unsigned o = 0;o < Os;++o)
                    {
                        const unsigned w = Is*o+i;
                        const T &dLdO = (*dO)[o];
                        dIv += dLdO * W[w];
                        W[w] -= dLdO * Iv;
                    }
                    B[i] -= dIv;
                }
                dI[i] = dIv;
            }

            dealloc_dO();
        }
    };

    template<typename T>
    struct OMPLayer : public CPULayer<T>
    {
        using Layer::I; using Layer::W; using Layer::B; using Layer::O;
        using Layer::dI; using Layer::dO; using Layer::LR;
        using Layer::Is; using Layer::Os;

        using CPULayer::CPULayer;
        using CPULayer::alloc_dI;
        using CPULayer::dealloc_dO;

        void forward() override
        {
            #pragma omp parallel for
            for(long i = 0;i < (long)Is;++i)
                I[i] = std::max<T>(T(0),I[i]+B[i]);
            #pragma omp parallel for
            for(long o = 0;o < (long)Os;++o)
            {
                T v(0);
                const unsigned wo = Is*(unsigned)o;
                for(unsigned i = 0;i < Is;++i)
                    v += I[i] * W[wo+i];
                (*O)[o] = v;
            }
        }
        /*
        Preconditions:
            - 'dO' already allocated
        Effects:
            - 'dO' deallocated
            - 'dI' allocated
        */
        void backward() override
        {
            alloc_dI();

            #pragma omp parallel for
            for(long o = 0;o < (long)Os;++o)
                (*dO)[o] *= LR;
            #pragma omp parallel for
            for(long i = 0;i < (long)Is;++i)
            {
                const T &Iv = I[i];
                T dIv(0);
                if(Iv != T(0))
                {
                    for(unsigned o = 0;o < Os;++o)
                    {
                        const unsigned w = Is*o+(unsigned)i;
                        const T dLdO = (*dO)[o];
                        dIv += dLdO * W[w];
                        W[w] -= dLdO * Iv;
                    }
                    B[i] -= dIv;
                }
                dI[i] = dIv;
            }

            dealloc_dO();
        }
    };

    template<typename T>
    __global__ void fwd_I(T *__restrict__ i,
                          const T *__restrict__ b,
                          const unsigned _Is)
    {
        const unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
        if(idx < _Is)
        {
            const T iv = i[idx] + b[idx];
            i[idx] = iv > T(0)? iv : T(0);
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

    /*
    NOTE:
        Max size for each layer should be maxThreadsPerBlock due to block reduction
    */
    template<typename T>
    struct GPULayer : public Layer<T>
    {
        using Layer::I; using Layer::W; using Layer::B; using Layer::O;
        using Layer::dI; using Layer::dO; using Layer::LR;
        using Layer::Is; using Layer::Os;
        T *d_I,*d_W,*d_B,**d_O;

        GPULayer(unsigned Is,unsigned Os,T **O,T **dO,T LR,T **d_O,
                 unsigned hostAllocMode_I = cudaHostAllocDefault,
                 unsigned hostAllocMode_W = cudaHostAllocDefault,
                 unsigned hostAllocMode_B = cudaHostAllocDefault)
            : Layer<T>(Is,Os,O,dO,LR),
              d_I(nullptr),d_W(nullptr),
              d_B(nullptr),d_O(d_O)
        {
            #ifdef DEBUG
            if(Is > (unsigned)GPU::properties.maxThreadsPerBlock)
                err("(Is=",Is,")>(maxThreadsPerBlock=",GPU::properties.maxThreadsPerBlock,')');
            if(Os > (unsigned)GPU::properties.maxThreadsPerBlock)
                err("(Os=",Os,")>(maxThreadsPerBlock=",GPU::properties.maxThreadsPerBlock,')');
            #endif // DEBUG
            alloc_I(hostAllocMode_I);
            alloc_W(hostAllocMode_W);
            alloc_B(hostAllocMode_B);
        }

        virtual ~GPULayer()
        {
            dealloc_I();
            dealloc_W();
            dealloc_B();
        }

        void alloc_I(unsigned mode) {GPU::allocHostPinned(&I,Is,mode);}
        void alloc_W(unsigned mode) {GPU::allocHostPinned(&W,Is*Os,mode);}
        void alloc_B(unsigned mode) {GPU::allocHostPinned(&B,Is,mode);}

        void dealloc_I() {GPU::destroyHostPinned((void*)I);}
        void dealloc_W() {GPU::destroyHostPinned((void*)W);}
        void dealloc_B() {GPU::destroyHostPinned((void*)B);}

        void alloc_dI(cudaStream_t stream) {GPU::allocDeviceMem(&dI,Is,stream);}
        // DEBUG: potentially unsafe
        void dealloc_dO(cudaStream_t stream) {GPU::destroyDeviceMem(dO,stream);}

        void alloc_d_I(cudaStream_t stream) {GPU::allocDeviceMem(&d_I,Is,stream);}
        void alloc_d_W(cudaStream_t stream) {GPU::allocDeviceMem(&d_W,(size_t)Is*Os,stream);}
        void alloc_d_B(cudaStream_t stream) {GPU::allocDeviceMem(&d_B,Is,stream);}

        // DEBUG: potentially unsafe
        void alloc_d_O(cudaStream_t stream) {GPU::allocDeviceMem(d_O,Os,stream);}
        // DEBUG: potentially unsafe
        void dealloc_d_O(cudaStream_t stream) {GPU::destroyDeviceMem(d_O,stream);}

        void dealloc_d_I(cudaStream_t stream) {GPU::destroyDeviceMem((void*)d_I,stream);}
        void dealloc_d_W(cudaStream_t stream) {GPU::destroyDeviceMem((void*)d_W,stream);}
        void dealloc_d_B(cudaStream_t stream) {GPU::destroyDeviceMem((void*)d_B,stream);}

        void transferH2D_I(cudaStream_t stream)
        {
            GPU::transfer<T,cudaMemcpyHostToDevice>(I,d_I,Is,stream);
        }
        void transferH2D_W(cudaStream_t stream)
        {
            GPU::transfer<T,cudaMemcpyHostToDevice>(W,d_W,(size_t)Is*Os,stream);
        }
        void transferH2D_B(cudaStream_t stream)
        {
            GPU::transfer<T,cudaMemcpyHostToDevice>(B,d_B,Is,stream);
        }
        void transferH2D_O(cudaStream_t stream)
        {
            GPU::transfer<T,cudaMemcpyHostToDevice>(*O,*d_O,Os,stream);
        }

        void transferD2H_I(cudaStream_t stream)
        {
            GPU::transfer<T,cudaMemcpyDeviceToHost>(d_I,I,Is,stream);
        }
        void transferD2H_W(cudaStream_t stream)
        {
            GPU::transfer<T,cudaMemcpyDeviceToHost>(d_W,W,(size_t)Is*Os,stream);
        }
        void transferD2H_B(cudaStream_t stream)
        {
            GPU::transfer<T,cudaMemcpyDeviceToHost>(d_B,B,Is,stream);
        }
        
        /*
        Preconditions:
            - 'I' already in device memory
        Effects:
            - 'I' removed from device memory
            - 'O' allocated to device memory
        */
        void forward(cudaStream_t stream)
        {
            // Allocate and transfer bias vector
            alloc_d_B(stream);
            transferH2D_B(stream);

            // Compute ReLU and bias
            {
                const unsigned maxBlock = GPU::properties.maxThreadsPerBlock;
                dim3 block(std::min<unsigned>(maxBlock,Is));
                dim3 grid((block.x-1+maxBlock)/maxBlock);

                fwd_I CONFIG4(grid,block,0,stream)(d_I,d_B,Is);
            }

            // Cleanup bias vector
            dealloc_d_B(stream);

            // Allocate and transfer weight matrix
            alloc_d_W(stream);
            transferH2D_W(stream);

            // Allocate ouput vector
            alloc_d_O(stream);

            // Compute 'I*W'
            {
                dim3 block(Is);
                dim3 grid(Os);

                fwd_main CONFIG4(
                    grid,block,
                    GPU::reduceSM<T>(Is,GPU::properties.warpSize),
                    stream
                )(d_I,d_W,*d_O,Is,Os);
            }

            // Cleanup weights
            dealloc_d_W(stream);

            // Transfer and clean up input
            transferD2H_I(stream);
            dealloc_d_I(stream);
        }
        void forward() override {forward(0);}

        /*
        Preconditions:
            - 'dO' already in device memory
        Effects:
            - 'dO' deallocated
            - 'dI' allocated in device memory
        */
        void backward(cudaStream_t stream)
        {
            // Scale by learning rate
            {
                const unsigned maxBlock = GPU::properties.maxThreadsPerBlock;
                dim3 block(std::min<unsigned>(maxBlock,Os));
                dim3 grid((block.x-1+maxBlock)/maxBlock);

                bkwd_dO CONFIG4(grid,block,0,stream)(*dO,LR,Os);
            }

            // Allocate and transfer input
            alloc_d_I(stream);
            transferH2D_I(stream);

            // Allocate and transfer bias
            alloc_d_B(stream);
            transferH2D_B(stream);

            // Allocate and transfer weights
            alloc_d_W(stream);
            transferH2D_W(stream);

            // Allocate dL/dI
            alloc_dI(stream);

            // Compute dL/dI, update weights, and update bias
            {
                dim3 block(Os);
                dim3 grid(Is);

                bkwd_main CONFIG4(
                    grid,block,
                    GPU::reduceSM<T>(Os,GPU::properties.warpSize),
                    stream
                )(*dO,d_I,d_W,d_B,dI,Is,Os);
            }

            // Cleanup dL/dO
            dealloc_dO(stream);

            // Transfer and cleanup weights
            transferD2H_W(stream);
            dealloc_d_W(stream);

            // Transfer and cleanup bias
            transferD2H_B(stream);
            dealloc_d_B(stream);

            // Cleanup input
            dealloc_d_I(stream);
        }
        void backward() override {backward(0);}
    };
}