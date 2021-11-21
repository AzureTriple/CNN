#pragma once

#include "intellisense_fix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <omp.h>
#include "GPU.cuh"

namespace convolve
{
    /*
    Note:
        Each matrix is indexed <row,column,channel> in decreasing order
    */
    template<typename T>
    struct Layer
    {
        T *I,*F,**O,*B,
          *dI,**dO,LR;
        unsigned Ir,Ic,
                 Fr,Fc,nF,
                 Or,Oc,Ch,
                 Pr,Pc,
                 Sr,Sc;

        Layer(unsigned Ir,unsigned Ic,
              unsigned Fr,unsigned Fc,unsigned nF,
              unsigned Or,unsigned Oc,unsigned Ch,
              unsigned Pr,unsigned Pc,
              unsigned Sr,unsigned Sc,
              T **O,T **dO,T LR)
            : Ir(Ir),Ic(Ic),
              Fr(Fr),Fc(Fc),nF(nF),
              Or(Or),Oc(Oc),Ch(Ch),
              Pr(Pr),Pc(Pc),
              Sr(Sr),Sc(Sc),
              O(O),dO(dO),LR(LR),
              I(nullptr),
              F(nullptr),
              B(nullptr),
              dI(nullptr)
        {
            #ifdef DEBUG
            if(!Ir)
                err("(Ir=",Ir,")==0");
            if(!Ic)
                err("(Ic=",Ic,")==0");
            if(!Fr)
                err("(Fr=",Fr,")==0");
            if(!Fc)
                err("(Fc=",Fc,")==0");
            if(!nF)
                err("(nF=",nF,")==0");
            if(!Or)
                err("(Or=",Or,")==0");
            if(!Oc)
                err("(Oc=",Oc,")==0");
            if(!Ch)
                err("(Ch=",Ch,")==0");
            if(!Pr)
                err("(Pr=",Pr,")==0");
            if(!Pc)
                err("(Pc=",Pc,")==0");
            if(!Sr)
                err("(Sr=",Sr,")==0");
            if(!Sc)
                err("(Sc=",Sc,")==0");
            if(LR < T(0))
                err("(LR=",LR,")<0");
            if((Or-1)*Sr != Ir - Fr + 2*Pr)
                err("[((Or=",Or,")-1)*(Sr=",Sr,")-(Ir=",Ir,")+(Fr=",Fr,")-2*(Pr=",Pr,")=",(long long)(Or-1)*Sr-Ir+Fr-2*Pr,"]!=0");
            if((Oc-1)*Sc != Ic - Fc + 2*Pc)
                err("[((Oc=",Oc,")-1)*(Sc=",Sc,")-(Ic=",Ic,")+(Fc=",Fc,")-2*(Pc=",Pc,")=",(long long)(Oc-1)*Sc-Ic+Fc-2*Pc,"]!=0");
            if(Fr <= Pr)
                err("(Fr=",Fr,")<=(Pr=",Pr,')');
            if(Fc <= Pc)
                err("(Fc=",Fc,")<=(Pc=",Pc,')');
            #endif // DEBUG
        }

        virtual ~Layer() {}

        virtual void alloc_I() = 0;

        virtual void dealloc_I() = 0;
        virtual void dealloc_F() = 0;
        virtual void dealloc_B() = 0;

        virtual void forward() = 0;
        virtual void backward() = 0;
    };

    template<typename T>
    struct CPULayer : public Layer<T>
    {
        using Layer::I ; using Layer::F ; using Layer::O ; using Layer::B;
        using Layer::dI; using Layer::dO; using Layer::LR;
        using Layer::Ir; using Layer::Ic;
        using Layer::Fr; using Layer::Fc; using Layer::nF;
        using Layer::Or; using Layer::Oc; using Layer::Ch;
        using Layer::Pr; using Layer::Pc;
        using Layer::Sr; using Layer::Sc;

        CPULayer(unsigned Ir,unsigned Ic,
                 unsigned Fr,unsigned Fc,unsigned nF,
                 unsigned Or,unsigned Oc,unsigned Ch,
                 unsigned Pr,unsigned Pc,
                 unsigned Sr,unsigned Sc,
                 T **O,T **dO,T LR)
            : Layer<T>(Ir,Ic,Fr,Fc,nF,Or,Oc,Ch,Pr,Pc,Sr,Sc,O,dO,LR)
        {
            alloc_I();
            alloc_F();
            alloc_B();
        }

        virtual ~CPULayer()
        {
            dealloc_I();
            dealloc_F();
            dealloc_B();
        }

        void alloc_I() {I = (T*)malloc(sizeof(T)*Ir*Ic*Ch);}
        void dealloc_I() {free((void*)I);}
        
        void alloc_F() {F = (T*)malloc(sizeof(T)*Fr*Fc*Ch*nF);}
        void dealloc_F() {free((void*)F);}

        void alloc_B() {B = (T*)malloc(sizeof(T)*nF);}
        void dealloc_B() {free((void*)B);}

        void alloc_dI() {dI = (T*)calloc((size_t)Ir*Ic*Ch,sizeof(T));}
        void dealloc_dI() {free((void*)dI);}

        // DEBUG: potentially unsafe
        void dealloc_dO() {free((void*)*dO);}
    };

    template<typename T>
    struct STCLayer : public CPULayer<T>
    {
        using Layer::I ; using Layer::F ; using Layer::O ; using Layer::B;
        using Layer::dI; using Layer::dO; using Layer::LR;
        using Layer::Ir; using Layer::Ic;
        using Layer::Fr; using Layer::Fc; using Layer::nF;
        using Layer::Or; using Layer::Oc; using Layer::Ch;
        using Layer::Pr; using Layer::Pc;
        using Layer::Sr; using Layer::Sc;

        using CPULayer::CPULayer;
        using CPULayer::alloc_dI;
        using CPULayer::dealloc_dO;

        T *dF;

        virtual ~STCLayer() {}

        void alloc_dF() {dF = (T*)calloc((size_t)Fr*Fc*Ch*nF,sizeof(T));}
        void dealloc_dF() {free((void*)dF);}

        void forward()
        {
            for(unsigned or = 0;or < Or;++or)
            {
                const unsigned _or = Oc*nF*or,
                               sor = Sr*or,
                            dprsor = Pr-sor,
                               fr0 = Pr > sor? dprsor : 0,
                               fr1 = std::min<unsigned>(Fr,Ir+dprsor);
                for(unsigned oc = 0;oc < Oc;++oc)
                {
                    const unsigned _oc = _or+nF*oc,
                                   soc = Sc*oc,
                                dpcsoc = Pc-soc,
                                   fc0 = Pc > soc? dpcsoc : 0,
                                   fc1 = std::min<unsigned>(Fc,Ic+dpcsoc);
                    for(unsigned f = 0;f < nF;++f)
                    {
                        const unsigned _och = _oc+f,
                                       _fch = Fr*Fc*Ch*f;
                        T o(B[f]);
                        for(unsigned fr = fr0;fr < fr1;++fr)
                        {
                            const unsigned _ir = Ic*Ch*(fr-dprsor),
                                           _fr = _fch+Fc*Ch*fr;
                            for(unsigned fc = fc0;fc < fc1;++fc)
                            {
                                const unsigned _ic = _ir+Ch*(fc-dpcsoc),
                                               _fc = _fr+Ch*fc;
                                for(unsigned ch = 0;ch < Ch;++ch)
                                    o += I[_ic+ch] * F[_fc+ch];
                            }
                        }
                        (*O)[_och] = std::max<T>(T(0),o);
                    }
                }
            }
        }

        /*
        Preconditions:
            - 'dO' allocated
        Effects:
            - 'dO' deallocated
            - 'dI' allocated

        Note:
            The loop structure of the serial algorithm differs from the parallel algorithms because
            the gradient calculations can be skipped where the ReLU output is zero. This means that
            more work can potentially be saved by following the forward algorithm and skipping when
            the output was zero.
        */
        void backward()
        {
            alloc_dI();
            alloc_dF();

            for(unsigned or = 0;or < Or;++or)
            {
                const unsigned _or = Oc*nF*or,
                               sor = Sr*or,
                            dprsor = Pr-sor,
                               fr0 = Pr > sor? dprsor : 0,
                               fr1 = std::min<unsigned>(Fr,Ir+dprsor);
                for(unsigned oc = 0;oc < Oc;++oc)
                {
                    const unsigned _oc = _or+nF*oc,
                                   soc = Sc*oc,
                                dpcsoc = Pc-soc,
                                   fc0 = Pc > soc? dpcsoc : 0,
                                   fc1 = std::min<unsigned>(Fc,Ic+dpcsoc);
                    for(unsigned f = 0;f < nF;++f)
                    {
                        const unsigned _och = _oc+f,
                                       _fch = Fr*Fc*Ch*f;
                        if((*O)[_och])
                        {
                            const T &dLdO = (*dO)[_och] * LR;
                            B[f] -= dLdO;
                            for(unsigned fr = fr0;fr < fr1;++fr)
                            {
                                const unsigned _ir = Ic*Ch*(fr-dprsor),
                                               _fr = _fch+Fc*Ch*fr;
                                for(unsigned fc = fc0;fc < fc1;++fc)
                                {
                                    const unsigned _ic = _ir+Ch*(fc-dpcsoc),
                                                   _fc = _fr+Ch*fc;
                                    for(unsigned ch = 0;ch < Ch;++ch)
                                    {
                                        const unsigned _ix = _ic+ch,
                                                       _fx = _fc+ch;
                                        dI[_ix] += dLdO * F[_fx];
                                        dF[_fx] += dLdO * I[_ix];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            dealloc_dO();
            for(unsigned f = 0;f < Fr*Fc*Ch*nF;++f)
                F[f] -= dF[f];
            dealloc_dF();
        }
    };

    template<typename T>
    struct OMPLayer : public CPULayer<T>
    {
        using Layer::I ; using Layer::F ; using Layer::O ; using Layer::B;
        using Layer::dI; using Layer::dO; using Layer::LR;
        using Layer::Ir; using Layer::Ic;
        using Layer::Fr; using Layer::Fc; using Layer::nF;
        using Layer::Or; using Layer::Oc; using Layer::Ch;
        using Layer::Pr; using Layer::Pc;
        using Layer::Sr; using Layer::Sc;

        using CPULayer::CPULayer;
        using CPULayer::alloc_dI;
        using CPULayer::dealloc_dO;

        virtual ~OMPLayer() {}

        void forward()
        {
            #pragma omp parallel for
            for(long long x = 0;x < (long long)nF*Or*Oc;++x)
            {
                const unsigned or = (unsigned)(x/nF/Oc),
                               oc = (unsigned)(x/nF)%Oc,
                                f = (unsigned)(x%nF),
                             
                              sor = Sr*or,
                              soc = Sc*oc,
                           dprsor = Pr-sor,
                           dpcsoc = Pc-soc,
                              fr0 = Pr > sor? dprsor : 0,
                              fc0 = Pc > soc? dpcsoc : 0,
                              fr1 = std::min<unsigned>(Fr,Ir+dprsor),
                              fc1 = std::min<unsigned>(Fc,Ic+dpcsoc),

                               ff = Fr*Fc*Ch*f;
                T local(B[f]);
                for(unsigned fr = fr0;fr < fr1;++fr)
                {
                    const unsigned _fr = ff+Fc*Ch*fr,
                                    ir = Ic*Ch*(fr-dprsor);
                    for(unsigned fc = fc0;fc < fc1;++fc)
                    {
                        const unsigned frc = _fr+Ch*fc,
                                       irc = ir+Ch*(fc-dpcsoc);
                        for(unsigned ch = 0;ch < Ch;++ch)
                            local += I[irc+ch] * F[frc+ch];
                    }
                }
                (*O)[x] = std::max<T>(T(0),local);
            }
        }

        /*
        Preconditions:
            - 'dO' allocated
        Effects:
            - 'dO' deallocated
            - 'dI' allocated
        */
        void backward()
        {
            // Pre-compute 'dL/dReLU'
            #pragma omp parallel for
            for(long long o = 0;o < (long long)Or*Oc*nF;++o)
                (*dO)[o] = (*O)[o]? (*dO)[o] * LR : T(0);

            // Update 'B'
            for(unsigned f = 0;f < nF;++f)
            {
                T b = B[f];
                #pragma omp parallel for reduction(-:b)
                for(long long o = 0;o < (long long)Or*Oc;++o)
                    b -= (*dO)[o*nF+f];
                B[f] = b;
            }

            // Allocate 'dI'
            alloc_dI();

            // Compute gradient w.r.t. 'I'
            #pragma omp parallel for
            for(long long i = 0;i < (long long)Ir*Ic*Ch;++i)
            {
                const unsigned ir = (unsigned)(i/Ch/Ic),
                               ic = (unsigned)((i/Ch)%Ic),
                               ch = (unsigned)(i%Ch),
                    
                // dL/dI can be modeled by the convolution of dL/dO with (F-P-1) padding, (S-1)
                // dilation and F mirrored across both rows and columns (effectively 180 degree
                // rotation).
                // 
                // The quantity 'S-1-((I-F+P)%S)' represents the first filter index where the value
                // is a part of the gradient computation, without accounting for padding. All other
                // values appear at intervals of 'S' from this index.
                // 
                // The quantity 'F-P-i-1' is the first filter index, only accounting for padding.
                //
                // The maximum of the two values is the absolute first filter index.

                             fmpr = Fr-Pr,
                             fmpc = Fc-Pc,
                               r1 = Sr-1-(ir-fmpr)%Sr,
                               c1 = Sc-1-(ic-fmpc)%Sc,
                               r0 = fmpr-1,
                               c0 = fmpc-1,
                               rx = std::max<unsigned>(r0>ir?r0-ir:0,r1),
                               cx = std::max<unsigned>(c0>ic?c0-ic:0,c1);
                T dldi(0);
                for(unsigned f = 0;f < nF;++f)
                {
                    const unsigned _ff = Fr*Fc*Ch*f+ch;
                    for(unsigned fr = rx;fr < Fr;fr += Sr)
                    {
                        const unsigned _fr = _ff+Fc*Ch*(Fr-1-fr),
                                       _or = Oc*nF*((fr+ir-fmpr)/Sr+1)+f;
                        for(unsigned fc = cx;fc < Fc;fc += Sc)
                            dldi += F[_fr+Ch*(Fc-1-fc)]*(*dO)[_or+nF*((fc+ic-fmpc)/Sc+1)];
                    }
                }
                dI[i] = dldi;
            }

            // Compute gradient w.r.t. 'F'
            #pragma omp parallel for
            for(long long x = 0;x < (long long)nF*Fr*Fc*Ch;++x)
            {
                const unsigned f = (unsigned)(x/Ch/Fc/Fr),
                              fr = (unsigned)((x/Ch/Fc)%Fr),
                              fc = (unsigned)((x/Ch)%Fc),
                              ch = (unsigned)(x%Ch),

                // dL/dF can be modeled by the convolution of I with original padding and dL/dO
                // with (S-1) dilation.
                // 
                // The quantity '(P-f+S-1)/S' represents the first dL/dO index where the value is a
                // part of the gradient computation. All other values appear at intervals of 'S'
                // from this index.
                // 
                // The quantity '(I-1+P-f)/S' is the last filter index.
                // 
                // The quantity 'o*S-P+f' is the input index corresponding to dL/dO index 'o'.

                           dprfr = Pr-fr,
                           dpcfc = Pc-fc,
                             or0 = Pr > fr? (dprfr+Sr-1)/Sr : 0,
                             oc0 = Pc > fc? (dpcfc+Sc-1)/Sc : 0,
                             or1 = std::min<unsigned>(Or,(Ir-1+dprfr)/Sr+1),
                             oc1 = std::min<unsigned>(Oc,(Ic-1+dpcfc)/Sc+1);

                T dldf(0);
                for(unsigned or = or0;or < or1;++or)
                {
                    const unsigned _or = Oc*nF*or+f,
                                    ir = Ic*Ch*(or*Sr-dprfr)+ch;
                    for(unsigned oc = oc0;oc < oc1;++oc)
                        dldf += I[ir+Ch*(oc*Sc-dpcfc)]*(*dO)[_or+nF*oc];
                }
                F[x] -= dldf;
            }

            // Cleanup 'dO'
            dealloc_dO();
        }
    };

    struct CONSTANTS
    {
        const unsigned Ir,Ic,
                       Fr,Fc,nF,
                       Or,Oc,Ch,
                       Pr,Pc,
                       Sr,Sc;
    };

    template<typename T>
    __global__ void fwdKernel(const T *__restrict__ i,
                              const T *__restrict__ f,
                              T *__restrict__ o,
                              const T *__restrict__ b,
                              const CONSTANTS C)
    {
        // Calculate indices
        const unsigned och = blockIdx.z,
                        or = blockIdx.y,
                        oc = blockIdx.x,
                        ch = threadIdx.z,
                        fr = threadIdx.y,
                        fc = threadIdx.x,
                 
                       sor = C.Sr * or,
                       soc = C.Sc * oc,
                    dprsor = C.Pr - sor,
                    dpcsoc = C.Pc - soc,
                       fr0 = C.Pr > sor? dprsor : 0,
                       fc0 = C.Pc > soc? dpcsoc : 0,
                       fr1 = umin(C.Fr,C.Ir+dprsor),
                       fc1 = umin(C.Fc,C.Ic+dpcsoc),

                        ir = fr-dprsor,
                        ic = fc-dpcsoc,

                        ix = C.Ic*C.Ch*ir+
                                  C.Ch*ic+
                                       ch,
                        fx = C.Fr*C.Fc*C.Ch*och+
                                  C.Fc*C.Ch*fr+
                                       C.Ch*fc+
                                            ch,
                        ox = C.Oc*C.nF*or+
                                  C.nF*oc+
                                       och;

        // Calculate local product
        T local(0);
        if(och < C.nF && fr0 <= fr && fr < fr1 && fc0 <= fc && fc < fc1)
            local = i[ix] * f[fx];

        // Sum and set output
        local = blockReduceSum(local);
        if(!(fr|fc|ch))
        {
            local += b[och];
            o[ox] = local > T(0)? local : T(0);
        }
    }
    template<typename T>
    __global__ void dLdB(T *__restrict__ dO,
                         const T *__restrict__ o,
                         T *__restrict__ b,
                         const CONSTANTS C,
                         const T C_LR)
    {
        const unsigned or = threadIdx.y,
                       oc = threadIdx.x,
                        f = blockIdx.x,
                       ox = C.Oc*C.nF*or+
                                 C.nF*oc+
                                      f;
        T dldb(0);
        if(ox < C.Or*C.Oc*C.nF)
            dldb = dO[ox] = o[ox]? dO[ox]*C_LR : T(0);
        dldb = blockReduceSum(dldb);
        if(!(or|oc)) b[f] -= dldb;
    }
    template<typename T>
    __global__ void dLdI(const T *__restrict__ dO,
                         T *__restrict__ dI,
                         const T *__restrict__ f,
                         const CONSTANTS C)
    {
        // Calculate indices
        const unsigned ch = blockIdx.z,
                       ir = blockIdx.y,
                       ic = blockIdx.x,

        // dL/dI can be modeled by the convolution of dL/dO with (F-P-1) padding, (S-1) dilation
        // and F mirrored across both rows and columns (effectively 180 degree rotation).
        // 
        // The quantity 'S-1-((I-F+P)%S)' represents the first filter index where the value is a
        // part of the gradient computation, without accounting for padding. All other values
        // appear at intervals of 'S' from this index.
        // 
        // The quantity 'F-P-i-1' is the first filter index, only accounting for padding.
        //
        // The maximum of the two values is the absolute first filter index.
            
                     fmpr = C.Fr-C.Pr,
                     fmpc = C.Fc-C.Pc,
                       r1 = C.Sr-1-(ir-fmpr)%C.Sr,
                       c1 = C.Sc-1-(ic-fmpc)%C.Sc,
                       r0 = fmpr-1,
                       c0 = fmpc-1,
                       rx = umax(r0>ir?r0-ir:0u,r1),
                       cx = umax(c0>ic?c0-ic:0u,c1),
            
                       ff = threadIdx.z,
                       fr = threadIdx.y*C.Sr+rx,
                       fc = threadIdx.x*C.Sc+cx,
            
                      _fx = C.Fr*C.Fc*C.Ch*ff+
                                 C.Fc*C.Ch*(C.Fr-1-fr)+
                                      C.Ch*(C.Fc-1-fc)+
                                           ch,
                      _ox = C.Oc*C.nF*((fr+ir-fmpr)/C.Sr+1)+
                                 C.nF*((fc+ic-fmpc)/C.Sc+1)+
                                      ff,
                      _ix = C.Ic*C.Ch*ir+
                                 C.Ch*ic+
                                      ch;
        T dldi(0);
        if(ff < blockDim.z && fr < C.Fr && fc < C.Fc)
            dldi = f[_fx] * dO[_ox];
        dldi = blockReduceSum(dldi);
        if(!(threadIdx.x|threadIdx.y|threadIdx.z))
            dI[_ix] = dldi;
    }
    template<typename T>
    __global__ void dLdF(const T *__restrict__ dO,
                         T *__restrict__ f,
                         const T *__restrict__ i,
                         const CONSTANTS C)
    {
        // Compute indices
        const unsigned ff = blockIdx.z,
                       fr = blockIdx.y,
                       fc = blockIdx.x/C.Ch,
                       ch = blockIdx.x%C.Ch,

        // dL/dF can be modeled by the convolution of I with original padding and dL/dO with (S-1)
        // dilation.
        // 
        // The quantity '(P-f+S-1)/S' represents the first dL/dO index where the value is a part of
        // the gradient computation. All other values appear at intervals of 'S' from this index.
        // 
        // The quantity '(I-1+P-f)/S' is the last filter index.
        // 
        // The quantity 'o*S-P+f' is the input index corresponding to dL/dO index 'o'.
            
                    dprfr = C.Pr-fr,
                    dpcfc = C.Pc-fc,
                      or0 = C.Pr > fr?(dprfr+C.Sr-1)/C.Sr : 0,
                      oc0 = C.Pc > fc?(dpcfc+C.Sc-1)/C.Sc : 0,
                      or1 = umin(C.Or,(C.Ir-1+dprfr)/C.Sr+1),
                      oc1 = umin(C.Oc,(C.Ic-1+dpcfc)/C.Sc+1),
            
                       or = threadIdx.y,
                       oc = threadIdx.x,
            
                      _ox = C.Oc*C.nF*or+
                                 C.nF*oc+
                                      ff,
                      _ix = C.Ic*C.Ch*(or*C.Sr-dprfr)+
                                 C.Ch*(oc*C.Sc-dpcfc)+
                                      ch,
                      _fx = C.Fr*C.Fc*C.Ch*ff+
                                 C.Fc*C.Ch*fr+
                                      C.Ch*fc+
                                           ch;
        T dldf(0);
        if(or0 <= or && or < or1 && oc0 <= oc && oc < oc1)
            dldf = i[_ix] * dO[_ox];
        dldf = blockReduceSum(dldf);
        if(!(or|oc))
            f[_fx] -= dldf;
    }
    /*
    NOTE:
        Filter volume (Fr*Fc*Ch) and output volume (Or*Oc*nF) should not exceed maxThreadsPerBlock
        due to reductions.
    */
    template<typename T>
    struct GPULayer : virtual public Layer<T> {
        using Layer::I ; using Layer::F ; using Layer::O ; using Layer::B;
        using Layer::dI; using Layer::dO; using Layer::LR;
        using Layer::Ir; using Layer::Ic;
        using Layer::Fr; using Layer::Fc; using Layer::nF;
        using Layer::Or; using Layer::Oc; using Layer::Ch;
        using Layer::Pr; using Layer::Pc;
        using Layer::Sr; using Layer::Sc;

        T *d_I,*d_F,**d_O,*d_B;

        GPULayer(unsigned Ir,unsigned Ic,
                 unsigned Fr,unsigned Fc,unsigned nF,
                 unsigned Or,unsigned Oc,unsigned Ch,
                 unsigned Pr,unsigned Pc,
                 unsigned Sr,unsigned Sc,
                 T **O,T **dO,T **d_O,T LR,
                 unsigned hostAllocMode_I = 0,
                 unsigned hostAllocMode_F = 0,
                 unsigned hostAllocMode_B = 0)
            : Layer<T>(Ir,Ic,Fr,Fc,nF,Or,Oc,Ch,Pr,Pc,Sr,Sc,O,dO,LR),
              d_I(nullptr),d_F(nullptr),d_O(d_O),d_B(nullptr)
        {
            #ifdef DEBUG
            if(Fr*Fc*Ch > (unsigned)GPU::properties.maxThreadsPerBlock)
                err("[(Fr=",Fr,")*(Fc=",Fc,")*(Ch=",Ch,")=",
                    Fr*Fc*Ch,"]>(maxThreadsPerBlock=",
                    GPU::properties.maxThreadsPerBlock,')');
            if(Or*Oc*Ch > (unsigned)GPU::properties.maxThreadsPerBlock)
                err("[(Or=",Or,")*(Oc=",Oc,")*(nF=",nF,")=",
                    Or*Oc*nF,"]>(maxThreadsPerBlock=",
                    GPU::properties.maxThreadsPerBlock,')');
            #endif // DEBUG
            alloc_I(hostAllocMode_I);
            alloc_F(hostAllocMode_F);
            alloc_B(hostAllocMode_B);
        }

        virtual ~GPULayer()
        {
            dealloc_I();
            dealloc_F();
            dealloc_B();
        }

        void alloc_I(unsigned mode) {GPU::allocHostPinned(&I,(size_t)Ir*Ic*Ch,mode);}
        void alloc_I() {alloc_I(cudaHostAllocDefault);}
        void dealloc_I() {GPU::destroyHostPinned(I);}

        void alloc_F(unsigned mode = cudaHostAllocDefault) {GPU::allocHostPinned(&F,(size_t)Fr*Fc*Ch*nF,mode);}
        void dealloc_F() {GPU::destroyHostPinned(F);}

        void alloc_B(unsigned mode = cudaHostAllocDefault) {GPU::allocHostPinned(&B,(size_t)nF,mode);}
        void dealloc_B() {GPU::destroyHostPinned(B);}

        void alloc_dI(cudaStream_t stream = 0) {GPU::allocDeviceMem(&dI,(size_t)Ir*Ic*Ch,stream);}
        void dealloc_dI(cudaStream_t stream) {GPU::destroyDeviceMem(dI,stream);}
        void dealloc_dI() {dealloc_dI(0);}

        void alloc_d_I(cudaStream_t stream = 0) {GPU::allocDeviceMem(&d_I,(size_t)Ir*Ic*Ch,stream);}
        void dealloc_d_I(cudaStream_t stream = 0) {GPU::destroyDeviceMem(d_I,stream);}

        void alloc_d_F(cudaStream_t stream = 0) {GPU::allocDeviceMem(&d_F,(size_t)Fr*Fc*Ch*nF,stream);}
        void dealloc_d_F(cudaStream_t stream = 0) {GPU::destroyDeviceMem(d_F,stream);}

        void alloc_d_B(cudaStream_t stream = 0) {GPU::allocDeviceMem(&d_B,(size_t)nF,stream);}
        void dealloc_d_B(cudaStream_t stream = 0) {GPU::destroyDeviceMem(d_B,stream);}

        // DEBUG: potentially unsafe
        void alloc_d_O(cudaStream_t stream = 0) {GPU::allocDeviceMem(d_O,(size_t)Or*Oc*nF,stream);}
        // DEBUG: potentially unsafe
        void dealloc_d_O(cudaStream_t stream = 0) {GPU::destroyDeviceMem(*d_O,stream);}
        // DEBUG: potentially unsafe
        void dealloc_dO(cudaStream_t stream = 0) {GPU::destroyDeviceMem(*dO,stream);}

        void transferH2D_I(cudaStream_t stream = 0)
        {
            GPU::transfer<T,cudaMemcpyHostToDevice>(I,d_I,(size_t)Ir*Ic*Ch,stream);
        }
        void transferD2H_I(cudaStream_t stream = 0)
        {
            GPU::transfer<T,cudaMemcpyDeviceToHost>(d_I,I,(size_t)Ir*Ic*Ch,stream);
        }

        void transferH2D_F(cudaStream_t stream = 0)
        {
            GPU::transfer<T,cudaMemcpyHostToDevice>(F,d_F,(size_t)Fr*Fc*Ch*nF,stream);
        }
        void transferD2H_F(cudaStream_t stream = 0)
        {
            GPU::transfer<T,cudaMemcpyDeviceToHost>(d_F,F,(size_t)Fr*Fc*Ch*nF,stream);
        }

        void transferH2D_O(cudaStream_t stream = 0)
        {
            GPU::transfer<T,cudaMemcpyHostToDevice>(*O,*d_O,(size_t)Or*Oc*nF,stream);
        }
        void transferD2H_O(cudaStream_t stream = 0)
        {
            GPU::transfer<T,cudaMemcpyDeviceToHost>(*d_O,*O,(size_t)Or*Oc*nF,stream);
        }

        void transferH2D_B(cudaStream_t stream = 0)
        {
            GPU::transfer<T,cudaMemcpyHostToDevice>(B,d_B,(size_t)nF,stream);
        }
        void transferD2H_B(cudaStream_t stream = 0)
        {
            GPU::transfer<T,cudaMemcpyDeviceToHost>(d_B,B,(size_t)nF,stream);
        }

        /*
        Preconditions:
            - 'I' already in device and host memory
        Effects:
            - 'I' removed from device memory
            - 'O' allocated to device and host memory
        */
        void forward(cudaStream_t stream)
        {
            // Allocate device memory
            alloc_d_F(stream);
            alloc_d_O(stream);
            alloc_d_B(stream);

            // Transfer filters and bias
            transferH2D_F(stream);
            transferH2D_B(stream);

            // Create dimensions
            dim3 block(Fc,Fr,Ch);
            dim3 grid(Oc,Or,nF);

            // Call kernel
            fwdKernel CONFIG4(
                grid,block,
                GPU::reduceSM<T>(Fr*Fc*Ch,GPU::properties.warpSize),
                stream
            )(d_I,d_F,*d_O,d_B,{Ir,Ic,Fr,Fc,nF,Or,Oc,Ch,Pr,Pc,Sr,Sc});

            // Transfer (but don't destroy) results
            transferD2H_O(stream);

            // Clean up memory
            dealloc_d_B(stream);
            dealloc_d_F(stream);
            dealloc_d_I(stream);
        }
        void forward() {forward(0);}
        
        /*
        Preconditions:
            - 'dO'&'O' already in device memory
        Effects:
            - 'dO'&'O' removed from device memory
            - 'dI'&'I' in device memory
        */
        void backward(cudaStream_t stream)
        {
            const unsigned warpsize = GPU::properties.warpSize;

            // Set learning rate and constant memory
            const T C_LR = LR;
            const CONSTANTS C {Ir,Ic,
                               Fr,Fc,nF,
                               Or,Oc,Ch,
                               Pr,Pc,
                               Sr,Sc};

            // Allocate and transfer bias to GPU
            alloc_d_B(stream);
            transferH2D_B(stream);
            
            // Update bias parameter and gradient w.r.t. ReLU
            {
                dim3 block(Oc,Or);
                dim3 grid(nF);

                dLdB CONFIG4(
                    grid,block,
                    GPU::reduceSM<T>(Oc*Or,warpsize),
                    stream
                )(*dO,*d_O,d_B,C,C_LR);
            }

            // Get bias back and clean up 'B'&'O'
            transferD2H_B(stream);
            dealloc_d_B(stream);
            dealloc_d_O(stream);

            // Allocate 'F'&'dI'
            alloc_d_F(stream);
            alloc_dI(stream);

            // Send 'F'
            transferH2D_F(stream);

            // Get gradient w.r.t. 'I'
            {
                dim3 grid(Ic,Ir,Ch);
                dim3 block(Fc,Fr,nF);
                const unsigned smSize = GPU::reduceSM<T>(Fc*Fr*nF,warpsize);
                    
                dLdI CONFIG4(
                    grid,block,
                    smSize,
                    stream
                )(*dO,dI,d_F,C);
            }

            // Allocate and send 'I'
            alloc_d_I(stream);
            transferH2D_I(stream);
            {
                // Gradient w.r.t. F
                dim3 grid(Fc*Ch,Fr,nF);
                dim3 block(Oc,Or);
                const unsigned smSize = GPU::reduceSM<T>(Oc*Or,warpsize);

                dLdF CONFIG4(
                    grid,block,
                    smSize,
                    stream
                )(*dO,d_F,d_I,C);
            }

            // Cleanup 'dO'&'I'
            dealloc_dO(stream);
            dealloc_d_I(stream);

            // Transfer and cleanup 'F'
            transferD2H_F(stream);
            dealloc_d_F(stream);
        }
        void backward() {backward(0);}
    };
}