#pragma once

#ifdef DEBUG
#include "debug.h"
#endif // DEBUG

#include "intellisense_fix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <omp.h>
#include "GPU.cuh"

namespace pool
{
    template<typename T>
    struct Layer
    {
        T *I,**O,*dI,**dO;
        unsigned Ir,Ic,Ich,
                 Or,Oc,Och,
                 Pr,Pc,Pch,*P;

        Layer(T **O,T **dO,
              unsigned Ir,unsigned Ic,unsigned Ich,
              unsigned Or,unsigned Oc,unsigned Och,
              unsigned Pr,unsigned Pc,unsigned Pch)
            : O(O),dO(dO),
              Ir(Ir),Ic(Ic),Ich(Ich),
              Or(Or),Oc(Oc),Och(Och),
              Pr(Pr),Pc(Pc),Pch(Pch),P(nullptr)
        {
            #ifdef DEBUG
            if(!Ir)
                err("(Ir=",Ir,")==0");
            if(!Ic)
                err("(Ic=",Ic,")==0");
            if(!Ich)
                err("(Ich=",Ich,")==0");
            if(!Or)
                err("(Or=",Or,")==0");
            if(!Oc)
                err("(Oc=",Oc,")==0");
            if(!Och)
                err("(Och=",Och,")==0");
            if(!Pr)
                err("(Pr=",Pr,")==0");
            if(!Pc)
                err("(Pc=",Pc,")==0");
            if(!Pch)
                err("(Pch=",Pch,")==0");
            if(Or > Ir)
                err("(Ir=",Ir,")>(Or=",Or,')');
            if(Oc > Ic)
                err("(Ic=",Ic,")>(Oc=",Oc,')');
            if(Och > Ich)
                err("(Ich=",Ich,")>(Och=",Och,')');
            if(Ir % Pr)
                err("(Ir=",Ir,")%(Pr=",Pr,")!=0");
            if(Ic % Pc)
                err("(Ic=",Ic,")%(Pc=",Pc,")!=0");
            if(Ich % Pch)
                err("(Ich=",Ich,")%(Pch=",Pch,")!=0");
            if(Ir / Pr != Or)
                err("(Ir=",Ir,")/(Pr=",Pr,")!=(Or=",Or,')');
            if(Ic / Pc != Oc)
                err("(Ic=",Ic,")/(Pc=",Pc,")!=(Oc=",Oc,')');
            if(Ich / Pch != Och)
                err("(Ich=",Ich,")/(Pch=",Pch,")!=(Och=",Och,')');
            #endif // DEBUG
        }

        virtual ~Layer() {}

        virtual void forward() = 0;
        virtual void backward() = 0;
    };

    template<typename T>
    struct CPULayer : public Layer<T>
    {
        using Layer::I; using Layer::dI; using Layer::dO;
        using Layer::Ir; using Layer::Ic; using Layer::Ich;
        using Layer::Or; using Layer::Oc; using Layer::Och;
        using Layer::P;

        void alloc_P() {P = (unsigned*)malloc(Or*Oc*Och*sizeof(unsigned));}
        void dealloc_P() {free(P);}
        
        void alloc_I() {I = (T*)malloc(sizeof(T)*Ir*Ic*Ich);}
        void dealloc_I() {free((void*)I);}
        
        /* DEBUG: Potentially unsafe */
        void alloc_dI() {dI = (T*)calloc((size_t)Ir*Ic*Ich,sizeof(T));}
        /* DEBUG: Potentially unsafe */
        void dealloc_dO() {free((void*)*dO);}

        CPULayer(T **O,T **dO,
                 unsigned Ir,unsigned Ic,unsigned Ich,
                 unsigned Or,unsigned Oc,unsigned Och,
                 unsigned Pr,unsigned Pc,unsigned Pch)
            : Layer<T>(O,dO,Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch)
        {
            alloc_P();
            alloc_I();
        }

        virtual ~CPULayer()
        {
            dealloc_P();
            dealloc_I();
        }
    };

    template<typename T>
    struct STCLayer : public CPULayer<T>
    {
        using Layer::I ; using Layer::O ;
        using Layer::dI; using Layer::dO;
        using Layer::Ir; using Layer::Ic; using Layer::Ich;
        using Layer::Or; using Layer::Oc; using Layer::Och;
        using Layer::Pr; using Layer::Pc; using Layer::Pch;
        using Layer::P;

        using CPULayer::CPULayer;
        using CPULayer::alloc_P;
        using CPULayer::dealloc_P;
        using CPULayer::alloc_dI;
        using CPULayer::dealloc_dO;

        virtual ~STCLayer() {}

        void forward() override
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
                        T max = I[_ich];
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
                                    const T &i = I[ich];
                                    if(max < i)
                                    {
                                        max = i;
                                        midx = Pc*Pch*pr+Pch*pc+pch;
                                    }
                                }
                            }
                        }
                        (*O)[_och] = max;
                        P[_och] = midx;
                    }
                }
            }
        }
        /*
        Preconditions:
            - 'dO' already in memory
        Effects:
            - 'dO' removed from memory
            - 'dI' allocated
        */
        void backward() override 
        {
            alloc_dI();
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
                        dI[ix] = (*dO)[_och];
                    }
                }
            }
            dealloc_dO();
        }
    };

    template<typename T>
    struct OMPLayer : public CPULayer<T> {
        using Layer::I; using Layer::O;
        using Layer::dI; using Layer::dO;
        using Layer::Ir; using Layer::Ic; using Layer::Ich;
        using Layer::Or; using Layer::Oc; using Layer::Och;
        using Layer::Pr; using Layer::Pc; using Layer::Pch;
        using Layer::P;

        using CPULayer::CPULayer;
        using CPULayer::alloc_P;
        using CPULayer::dealloc_P;
        using CPULayer::alloc_dI;
        using CPULayer::dealloc_dO;

        virtual ~OMPLayer() {}

        void forward() override
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
                
                T max = I[_ich];
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
                            const T &i = I[ich];
                            if(max < i)
                            {
                                max = i;
                                midx = Pc*Pch*pr+Pch*pc+pch;
                            }
                        }
                    }
                }
                (*O)[_och] = max;
                P[_och] = midx;
            }
        }
        /*
        Preconditions:
            - 'dO' already in memory
        Effects:
            - 'dO' removed from memory
            - 'dI' allocated
        */
        void backward() override 
        {
            alloc_dI();
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
                dI[i] = (*dO)[o];
            }
            dealloc_dO();
        }
    };

    struct CONSTANTS {
        const unsigned Ir,Ic,Ich,
                       Or,Oc,Och,
                       Pr,Pc,Pch;
    };

    template<typename T>
    __global__ void fwdKernel(const T *__restrict__ I,
                              T *__restrict__ O,
                              unsigned *__restrict__ P,
                              const CONSTANTS C)
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
        pair<T> v {I[_i],0};
        if(pr < C.Pr && pc < C.Pc && pch < C.Pch)
            v = {I[i],p};
        v = blockReduceMax(v);
        if(!(pr|pc|pch))
        {
            O[_o] = v.v;
            P[_o] = v.k;
        }
    }
    template<typename T>
    __global__ void bkwdKernel(T *__restrict__ dI,
                               const T *__restrict__ dO,
                               const unsigned *__restrict__ P,
                               const CONSTANTS C)
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
    /*
    NOTE:
        (Pr*Pc*Pch) should not exceed maxThreadsPerBlock due to
        reductions.
    */
    template<typename T>
    struct GPULayer : public Layer<T>
    {
        using Layer::I; using Layer::O;
        using Layer::dI; using Layer::dO;
        using Layer::Ir; using Layer::Ic; using Layer::Ich;
        using Layer::Or; using Layer::Oc; using Layer::Och;
        using Layer::Pr; using Layer::Pc; using Layer::Pch;
        using Layer::P;

        T *h_I,**h_O;
        unsigned *d_P;

        void alloc_dI(cudaStream_t stream) {GPU::allocDeviceMem(&dI,Ir*Ic*Ich,stream);}
        /* DEBUG: Potentially unsafe */
        void dealloc_dO(cudaStream_t stream) {GPU::destroyDeviceMem(*dO,stream);}

        void alloc_P(unsigned mode) {GPU::allocHostPinned(&P,Or*Oc*Och,mode);}
        void dealloc_P() {GPU::destroyHostPinned(P);}
        void alloc_d_P(cudaStream_t stream) {GPU::allocDeviceMem(&d_P,Or*Oc*Och,stream);}
        void dealloc_d_P(cudaStream_t stream) {GPU::destroyDeviceMem(d_P,stream);}
        void transferH2D_P(cudaStream_t stream)
        {
            GPU::transfer<unsigned,cudaMemcpyHostToDevice>(P,d_P,Or*Oc*Och,stream);
        }
        void transferD2H_P(cudaStream_t stream)
        {
            GPU::transfer<unsigned,cudaMemcpyDeviceToHost>(d_P,P,Or*Oc*Och,stream);
        }
        
        void alloc_I(cudaStream_t stream) {GPU::allocDeviceMem(&I,Ir*Ic*Ich,stream);}
        void dealloc_I(cudaStream_t stream) {GPU::destroyDeviceMem(I,stream);}

        void alloc_h_I(unsigned mode) {GPU::allocHostPinned(&h_I,Ir*Ic*Ich,mode);}
        void dealloc_h_I() {GPU::destroyHostPinned(h_I);}

        /* DEBUG: Potentially unsafe */
        void alloc_O(cudaStream_t stream) {GPU::allocDeviceMem(O,Or*Oc*Och,stream);}
        void transferD2H_O(cudaStream_t stream)
        {
            GPU::transfer<T,cudaMemcpyDeviceToHost>(*O,*h_O,Or*Oc*Och,stream);
        }

        GPULayer(T **O,T **dO,T **h_O,
                 unsigned Ir,unsigned Ic,unsigned Ich,
                 unsigned Or,unsigned Oc,unsigned Och,
                 unsigned Pr,unsigned Pc,unsigned Pch,
                 unsigned pMode = cudaHostAllocDefault,
                 unsigned iMode = cudaHostAllocDefault)
            : Layer<T>(O,dO,Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch),
              h_O(h_O)
        {
            #ifdef DEBUG
            #endif // DEBUG
            alloc_P(pMode);
            alloc_h_I(iMode);
        }

        virtual ~GPULayer()
        {
            dealloc_P();
            dealloc_h_I();
        }

        /*
        Preconditions:
            - 'I' already in device memory
        Effects:
            - 'I' removed from device memory
            - 'O' allocated in device memory
        */
        void forward(cudaStream_t stream)
        {
            // Allocate 'P'&'O'
            alloc_d_P(stream);
            alloc_O(stream);
            
            // Start kernel
            dim3 block(Pc,Pr,Pch);
            dim3 grid(Oc,Or,Och);
            fwdKernel CONFIG4(
                grid,block,
                GPU::reduceSM<pair<T>>(Pr*Pc*Pch,(unsigned)GPU::properties.warpSize),
                stream
            )(I,*O,d_P,{Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch});

            // Cleanup 'I'
            dealloc_I(stream);

            // Transfer and cleanup 'P'
            transferD2H_P(stream);
            dealloc_d_P(stream);

            // Transfer 'O'
            transferD2H_O(stream);
        }
        void forward() override {forward(0);}

        /*
        Preconditions:
            - 'dO' allocated in device memory
        Effects:
            - 'dO' removed from device memory
            - 'dI' allocated in device memory
        */
        void backward(cudaStream_t stream)
        {
            // Allocate 'dI'
            alloc_dI(stream);

            // Allocate and transfer 'P'
            alloc_d_P(stream);
            transferH2D_P(stream);

            // Start kernel
            dim3 block(std::min<unsigned>((unsigned)GPU::properties.maxThreadsPerBlock,Or*Oc*Och));
            dim3 grid(std::max<unsigned>(1,(Or*Oc*Och-1+block.x)/block.x));
            bkwdKernel CONFIG4(grid,block,0,stream)(dI,*dO,d_P,{Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch});

            // Cleanup 'P'&'dO'
            dealloc_d_P(stream);
            dealloc_dO(stream);
        }
        void backward() override {backward(0);}
    };
}