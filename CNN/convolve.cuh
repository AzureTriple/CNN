#pragma once

#include "layer.h"
#include "GPU.cuh"

namespace conv
{
    using layer::Layer;

    template<typename T>
    struct ConvLayer : public Layer<T>
    {
        T **I,*F,*B,**dI;
        const T LR;
        const unsigned Ir,Ic,Ch,
                       Or,Oc,nF,
                       Fr,Fc,
                       Pr,Pc,
                       Sr,Sc;

        ConvLayer(const unsigned Or,const unsigned Oc,const unsigned nF,
                  const unsigned Fr,const unsigned Fc,const unsigned Ch,
                  const unsigned Pr,const unsigned Pc,
                  const unsigned Sr,const unsigned Sc,const T LR,
                  T **I,T **dI)
            : Layer<T>(),
              Or(Or),Oc(Oc),nF(nF),
              Fr(Fr),Fc(Fc),
              Pr(Pr),Pc(Pc),Sr(Sr),Sc(Sc),
              Ir((Or-1)*Sr-2*Pr+Fr),Ic((Oc-1)*Sc-2*Pc+Fc),Ch(Ch),
              I(I),dI(dI),LR(LR),
              F(nullptr),B(nullptr) {}

        virtual ~ConvLayer() {}

        void init();
    };

    template<typename T>
    struct STCLayer : public ConvLayer<T>
    {
        using ConvLayer::I; using ConvLayer::Ir; using ConvLayer::Ic; using ConvLayer::Ch;
        using ConvLayer::F; using ConvLayer::Fr; using ConvLayer::Fc;
        using ConvLayer::B;
        using Layer::O; using ConvLayer::Or; using ConvLayer::Oc; using ConvLayer::nF;
        using ConvLayer::Sr; using ConvLayer::Sc;
        using ConvLayer::Pr; using ConvLayer::Pc;
        using ConvLayer::dI; using Layer::dO;
        using ConvLayer::LR;

        STCLayer(const unsigned Or,const unsigned Oc,const unsigned nF,
                 const unsigned Fr,const unsigned Fc,const unsigned Ch,
                 const unsigned Pr,const unsigned Pc,
                 const unsigned Sr,const unsigned Sc,const T LR,
                 T **I,T **dI)
            : ConvLayer<T>(Or,Oc,nF,Fr,Fc,Ch,Pr,Pc,Sr,Sc,LR,I,dI)
        {
            F = new T[nF*Fr*Fc*Ch];
            B = new T[nF];
            O = new T[Or*Oc*nF];
        }
        ~STCLayer() {delete[] F,B,O;}

        void forward();
        void backward();
    };

    template<typename T>
    struct OMPLayer : public ConvLayer<T>
    {
        using ConvLayer::I; using ConvLayer::Ir; using ConvLayer::Ic; using ConvLayer::Ch;
        using ConvLayer::F; using ConvLayer::Fr; using ConvLayer::Fc;
        using ConvLayer::B;
        using Layer::O; using ConvLayer::Or; using ConvLayer::Oc; using ConvLayer::nF;
        using ConvLayer::Sr; using ConvLayer::Sc;
        using ConvLayer::Pr; using ConvLayer::Pc;
        using ConvLayer::dI; using Layer::dO;
        using ConvLayer::LR;

        OMPLayer(const unsigned Or,const unsigned Oc,const unsigned nF,
                 const unsigned Fr,const unsigned Fc,const unsigned Ch,
                 const unsigned Pr,const unsigned Pc,
                 const unsigned Sr,const unsigned Sc,const T LR,
                 T **I,T **dI)
            : ConvLayer<T>(Or,Oc,nF,Fr,Fc,Ch,Pr,Pc,Sr,Sc,LR,I,dI)
        {
            F = new T[nF*Fr*Fc*Ch];
            B = new T[nF];
            O = new T[Or*Oc*nF];
        }
        ~OMPLayer() {delete[] F,B,O;}

        void forward();
        void backward();
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
    struct GPULayer : public ConvLayer<T>
    {
        using ConvLayer::I; using ConvLayer::Ir; using ConvLayer::Ic; using ConvLayer::Ch;
        using ConvLayer::F; using ConvLayer::Fr; using ConvLayer::Fc;
        using ConvLayer::B;
        using Layer::O; using ConvLayer::Or; using ConvLayer::Oc; using ConvLayer::nF;
        using ConvLayer::Sr; using ConvLayer::Sc;
        using ConvLayer::Pr; using ConvLayer::Pc;
        using ConvLayer::dI; using Layer::dO;
        using ConvLayer::LR;

        T **d_I,*d_O;
        cudaStream_t stream;

        GPULayer(const unsigned Or,const unsigned Oc,const unsigned nF,
                 const unsigned Fr,const unsigned Fc,const unsigned Ch,
                 const unsigned Pr,const unsigned Pc,
                 const unsigned Sr,const unsigned Sc,const T LR,
                 T **I,T **dI,cudaStream_t stream,T **d_I)
            : ConvLayer<T>(Or,Oc,nF,Fr,Fc,Ch,Pr,Pc,Sr,Sc,LR,I,dI),
              stream(stream),d_I(d_I),d_O(nullptr)
        {
            GPU::allocHostPinned(&F,nF*Fr*Fc*Ch);
            GPU::allocHostPinned(&B,nF);
            GPU::allocHostPinned(&O,Or*Oc*nF);
        }
        ~GPULayer()
        {
            GPU::destroyHostPinned(F);
            GPU::destroyHostPinned(B);
            GPU::destroyHostPinned(O);
        }

        void forward();
        void backward();
    };
}

#include "intellisense_fix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <omp.h>
#include <random>

template<typename T>
void conv::ConvLayer<T>::init()
{
    static std::default_random_engine generator;
    const unsigned nI = Fr*Fc*Ch;
    std::normal_distribution<double> distribution(0,std::sqrt(2./nI));
    for(unsigned f = 0;f < nF*nI;++f)
        F[f] = distribution(generator);
}

template<typename T>
void conv::STCLayer<T>::forward()
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
                            o += (*I)[_ic+ch] * F[_fc+ch];
                    }
                }
                O[_och] = std::max<T>(T(0),o);
            }
        }
    }
}

template<typename T>
void conv::STCLayer<T>::backward()
{
    *dI = new T[Ir*Ic*Ch]();
    T *dF = new T[nF*Fr*Fc*Ch]();

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
                if(O[_och])
                {
                    const T &dLdO = dO[_och] * LR;
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
                                (*dI)[_ix] += dLdO * F[_fx];
                                dF[_fx] += dLdO * (*I)[_ix];
                            }
                        }
                    }
                }
            }
        }
    }
            
    delete[] dO;
    dO = nullptr;
    for(unsigned f = 0;f < Fr*Fc*Ch*nF;++f)
        F[f] -= dF[f];
    delete[] dF;
}

template<typename T>
void conv::OMPLayer<T>::forward()
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
                    local += (*I)[irc+ch]*F[frc+ch];
            }
        }
        O[x] = std::max<T>(T(0),local);
    }
}

template<typename T>
void conv::OMPLayer<T>::backward()
{
    // Pre-compute 'dL/dReLU'
    #pragma omp parallel for
    for(long long o = 0;o < (long long)Or*Oc*nF;++o)
        dO[o] = O[o]? dO[o] * LR : T(0);

    // Update 'B'
    for(unsigned f = 0;f < nF;++f)
    {
        T b = B[f];
        #pragma omp parallel for reduction(-:b)
        for(long long o = 0;o < (long long)Or*Oc;++o)
            b -= dO[o*nF+f];
        B[f] = b;
    }

    // Allocate 'dI'
    *dI = new T[Ir*Ic*Ch];

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
                    dldi += F[_fr+Ch*(Fc-1-fc)]*dO[_or+nF*((fc+ic-fmpc)/Sc+1)];
            }
        }
        (*dI)[i] = dldi;
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
                dldf += (*I)[ir+Ch*(oc*Sc-dpcfc)]*dO[_or+nF*oc];
        }
        F[x] -= dldf;
    }

    // Cleanup 'dO'
    delete[] dO;
    dO = nullptr;
}

template<typename T>
__global__ void fwdKernel(const T *__restrict__ i,
                          const T *__restrict__ f,
                          T *__restrict__ o,
                          const T *__restrict__ b,
                          const conv::CONSTANTS C)
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
__global__ void dLdB(const T *__restrict__ dO,
                     const T *__restrict__ o,
                     T *__restrict__ b,
                     const conv::CONSTANTS C,
                     const T LR)
{
    const unsigned or = threadIdx.y,
                   oc = threadIdx.x,
                    f = blockIdx.x,
                   ox = C.Oc*C.nF*or+
                             C.nF*oc+
                                  f;
    T dldb(0);
    //if(ox < C.Or*C.Oc*C.nF && o[ox])
    if(IN_BOUNDS(x) && IN_BOUNDS(y) && o[ox])
        dldb = dO[ox];
    dldb = blockReduceSum(dldb);
    if(!(or|oc)) b[f] -= dldb * LR;
}
template<typename T>
__global__ void dLdI(const T *__restrict__ dO,
                     T *__restrict__ dI,
                     const T *__restrict__ f,
                     const conv::CONSTANTS C)
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
    if(_fx < C.nF*C.Fr*C.Fc*C.Ch && _ox < C.Or*C.Oc*C.nF &&
       ff < blockDim.z && fr < C.Fr && fc < C.Fc)
        dldi = f[_fx] * dO[_ox];
    dldi = blockReduceSum(dldi);
    if(!(threadIdx.x|threadIdx.y|threadIdx.z))
        dI[_ix] = dldi;
}
template<typename T>
__global__ void dLdF(const T *__restrict__ dO,
                     T *__restrict__ f,
                     const T *__restrict__ i,
                     const conv::CONSTANTS C,
                     const T LR)
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
        f[_fx] -= dldf * LR;
}

/*
Preconditions:
    - 'I' already in device and host memory
Effects:
    - 'I' removed from device memory
    - 'O' allocated to device and host memory
*/
template<typename T>
void conv::GPULayer<T>::forward()
{
    // Allocate device memory
    T *d_F,*d_B;
    GPU::allocTransfer(F,&d_F,nF*Fr*Fc*Ch,stream);
    GPU::allocTransfer(B,&d_B,nF,stream);
    GPU::allocDeviceMem(&d_O,Or*Oc*nF,stream);

    // Create dimensions
    dim3 block(Fc,Fr,Ch);
    dim3 grid(Oc,Or,nF);

    // Call kernel
    const CONSTANTS C{Ir,Ic,
                      Fr,Fc,nF,
                      Or,Oc,Ch,
                      Pr,Pc,
                      Sr,Sc};
    fwdKernel CONFIG4(
        grid,block,
        REDUCE_SM(Fr*Fc*Ch,T),
        stream
    )(*d_I,d_F,d_O,d_B,C);

    // Transfer (but don't destroy) results
    GPU::transfer<T,cudaMemcpyDeviceToHost>(d_O,O,Or*Oc*nF,stream);

    // Clean up memory
    GPU::destroyDeviceMem(d_B,stream);
    GPU::destroyDeviceMem(d_F,stream);
    GPU::destroyDeviceMem(*d_I,stream);
    *d_I = nullptr;
}

/*
Preconditions:
    - 'dO' already in device memory
Effects:
    - 'dO' removed from device memory
    - 'dI' in device memory
*/
template<typename T>
void conv::GPULayer<T>::backward()
{
    // Set constants
    const CONSTANTS C {Ir,Ic,
                       Fr,Fc,nF,
                       Or,Oc,Ch,
                       Pr,Pc,
                       Sr,Sc};

    // Allocate and transfer 'B'&'O'
    T *d_B;
    GPU::allocTransfer(B,&d_B,nF,stream);
    GPU::allocTransfer(O,&d_O,Or*Oc*nF,stream);
            
    // Update bias parameter and gradient w.r.t. ReLU
    {
        dim3 block(Oc,Or);
        dim3 grid(nF);

        dLdB CONFIG4(
            grid,block,
            REDUCE_SM(Oc*Or,T),
            stream
        )(dO,d_O,d_B,C,LR);
    }

    // Get bias back and clean up 'B'&'O'
    GPU::destroyTransfer(d_B,B,nF,stream);
    GPU::destroyDeviceMem(d_O,stream);
    d_O = nullptr;

    // Allocate 'F'&'dI'
    T *d_F;
    GPU::allocTransfer(F,&d_F,nF*Fr*Fc*Ch,stream);
    GPU::allocDeviceMem(dI,Ir*Ic*Ch,stream);

    // Get gradient w.r.t. 'I'
    {
        dim3 grid(Ic,Ir,Ch);
        dim3 block(Fc,Fr,nF);
                    
        dLdI CONFIG4(
            grid,block,
            REDUCE_SM(Fr*Fc*nF,T),
            stream
        )(dO,*dI,d_F,C);
    }

    // Allocate and send 'I'
    GPU::allocTransfer(*I,d_I,Ir*Ic*Ch,stream);

    {
        // Gradient w.r.t. F
        dim3 grid(Fc*Ch,Fr,nF);
        dim3 block(Oc,Or);

        dLdF CONFIG4(
            grid,block,
            REDUCE_SM(Or*Oc,T),
            stream
        )(dO,d_F,*d_I,C,LR);
    }

    // Cleanup 'dO'&'I'
    GPU::destroyDeviceMem(dO,stream);
    dO = nullptr;
    GPU::destroyDeviceMem(*d_I,stream);
    *d_I = nullptr;

    // Transfer and cleanup 'F'
    GPU::destroyTransfer(d_F,F,nF*Fr*Fc*Ch,stream);
}