#ifdef DEBUG
#include "debug.h"
#endif // DEBUG
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <algorithm>
#include "intellisense_fix.h"
#include "GPU.cuh"
#include "convolve.cuh"

void convolve::convolveCPU(Matrix &I,Matrix &F,Matrix &O,
                           unsigned Sr,unsigned Sc,
                           unsigned Pr,unsigned Pc,
                           double bias)
{
    #ifdef DEBUG
    if((I.r + 2*Pr - F.r) % Sr)
        err("[(I.r=",I.r,")+2*(Pr=",Pr,")-(F.r=",F.r,")]%(Sr=",Sr,")=(",(I.r+2*Pr-F.r)%Sr,"!=0)");
    if((I.c + 2*Pc - F.c) % Sc)
        err("[(I.c=",I.c,")+2*(Pc=",Pc,")-(F.c=",F.c,")]%(Sc=",Sc,")=(",(I.c+2*Pc-F.c)%Sc,"!=0)");
    if((I.r - F.r + 2*Pr)/Sr != O.r)
        err("[(I.r=",I.r,")-(F.r=",F.r,")+2*(Pr=",Pr,")]/(Sr=",Sr,")=[",(I.r-F.r+2*Pr)/Sr,"!=(O.r=",O.r,")]");
    if((I.c - F.c + 2*Pc)/Sc != O.c)
        err("[(I.c=",I.c,")-(F.c=",F.c,")+2*(Pc=",Pc,")]/(Sc=",Sc,")=[",(I.c-F.c+2*Pc)/Sc,"!=(O.c=",O.c,")]");
    if(I.ch != F.ch)
        err("(I.ch=",I.ch,")!=(F.ch=",F.ch,')');
    if(I.ch != O.ch)
        err("(I.ch=",I.ch,")!=(O.ch=",O.ch,')');
    if(F.ch != O.ch)
        err("(F.ch=",F.ch,")!=(O.ch=",O.ch,')');
    #endif
    double *Id = I.data,
           *Fd = F.data,
           *Od = O.data;
    for(unsigned ch = 0;ch < O.ch;++ch)
    {
        const unsigned Ich = I.r * I.c * ch,
                       Fch = F.r * F.c * ch,
                       Och = O.r * O.c * ch;
        for(unsigned Or = 0;Or < O.r;++Or)
        {
            const unsigned SOr = Or * Sr,
                           fr0 = Pr > SOr? Pr-SOr : 0U,
                           fr1 = std::min(F.r,I.r+Pr-SOr),
                          Orch = Och + O.c * Or;
            for(unsigned Oc = 0;Oc < O.c;++Oc)
            {
                const unsigned SOc = Oc * Sc,
                               fc0 = Pc > SOc? Pc-SOc : 0U,
                               fc1 = std::min(F.c,I.c+Pc-SOc);
                double &Ochrc = Od[Orch + Oc];
                for(unsigned fr = fr0;fr < fr1;++fr)
                {
                    const unsigned Irch = Ich + I.c * (SOr + fr - Pr),
                                   Frch = Fch + F.c * fr;
                    for(unsigned fc = fc0;fc < fc1;++fc)
                    {
                        const unsigned Ichrc = Irch + SOc + fc - Pc,
                                       Fchrc = Frch + fc;
                        Ochrc += Id[Ichrc] * Fd[Fchrc];
                    }
                }
                Ochrc = std::max<double>(0.,Ochrc+bias);
            }
        }
    }
}
void convolve::convolveOMP(Matrix &I,Matrix &F,Matrix &O,
                           unsigned Sr,unsigned Sc,
                           unsigned Pr,unsigned Pc,
                           double bias)
{
    #ifdef DEBUG
    if((I.r + 2*Pr - F.r) % Sr)
        err("[(I.r=",I.r,")+2*(Pr=",Pr,")-(F.r=",F.r,")]%(Sr=",Sr,")=(",(I.r+2*Pr-F.r)%Sr,"!=0)");
    if((I.c + 2*Pc - F.c) % Sc)
        err("[(I.c=",I.c,")+2*(Pc=",Pc,")-(F.c=",F.c,")]%(Sc=",Sc,")=(",(I.c+2*Pc-F.c)%Sc,"!=0)");
    if((I.r - F.r + 2*Pr)/Sr != O.r)
        err("[(I.r=",I.r,")-(F.r=",F.r,")+2*(Pr=",Pr,")]/(Sr=",Sr,")=[",(I.r-F.r+2*Pr)/Sr,"!=(O.r=",O.r,")]");
    if((I.c - F.c + 2*Pc)/Sc != O.c)
        err("[(I.c=",I.c,")-(F.c=",F.c,")+2*(Pc=",Pc,")]/(Sc=",Sc,")=[",(I.c-F.c+2*Pc)/Sc,"!=(O.c=",O.c,")]");
    if(I.ch != F.ch)
        err("(I.ch=",I.ch,")!=(F.ch=",F.ch,')');
    if(I.ch != O.ch)
        err("(I.ch=",I.ch,")!=(O.ch=",O.ch,')');
    if(F.ch != O.ch)
        err("(F.ch=",F.ch,")!=(O.ch=",O.ch,')');
    #endif
    double *Id = I.data,
           *Fd = F.data,
           *Od = O.data;
    const unsigned _Ir = I.r,_Ic = I.c,
                   _Fr = F.r,_Fc = F.c,
                   _Or = O.r,_Oc = O.c,
                   _Ch = O.ch;
    #pragma omp parallel for
    for(long Ox = 0;Ox < (long)(_Ch*_Or*_Oc);++Ox)
    {
        const unsigned ch = Ox/_Oc/_Or,
                       Or = (Ox/_Oc)%_Or,
                       Oc = Ox%_Oc,
                      Ich = _Ir * _Ic * ch,
                      Fch = _Fr * _Fc * ch,
                      Och = _Or * _Oc * ch,
                      SOr = Or * Sr,
                      fr0 = Pr > SOr? Pr-SOr : 0U,
                      fr1 = std::min(_Fr,_Ir+Pr-SOr),
                     Orch = Och + _Oc * Or,
                      SOc = Oc * Sc,
                      fc0 = Pc > SOc? Pc-SOc : 0U,
                      fc1 = std::min(_Fc,_Ic+Pc-SOc);
        double &Ochrc = Od[Orch + Oc];
        for(unsigned fr = fr0;fr < fr1;++fr)
        {
            const unsigned Irch = Ich + _Ic * (SOr + fr - Pr),
                           Frch = Fch + _Fc * fr;
            for(unsigned fc = fc0;fc < fc1;++fc)
            {
                const unsigned Ichrc = Irch + SOc + fc - Pc,
                               Fchrc = Frch + fc;
                Ochrc += Id[Ichrc] * Fd[Fchrc];
            }
        }
        Ochrc = std::max<double>(0.,Ochrc+bias);
    }
}

void convolve::convolveConstants(unsigned Ir,unsigned Ic,
                                 unsigned Fr,unsigned Fc,
                                 unsigned Or,unsigned Oc,
                                 unsigned Ch,unsigned Sr,
                                 unsigned Sc,unsigned Pr,
                                 unsigned Pc,cudaStream_t stream)
{
    #ifdef DEBUG
    if((I.r + 2*Pr - F.r) % Sr)
        err("[(I.r=",I.r,")+2*(Pr=",Pr,")-(F.r=",F.r,")]%(Sr=",Sr,")=(",(I.r+2*Pr-F.r)%Sr,"!=0)");
    if((I.c + 2*Pc - F.c) % Sc)
        err("[(I.c=",I.c,")+2*(Pc=",Pc,")-(F.c=",F.c,")]%(Sc=",Sc,")=(",(I.c+2*Pc-F.c)%Sc,"!=0)");
    if((I.r - F.r + 2*Pr)/Sr != O.r)
        err("[(I.r=",I.r,")-(F.r=",F.r,")+2*(Pr=",Pr,")]/(Sr=",Sr,")=[",(I.r-F.r+2*Pr)/Sr,"!=(O.r=",O.r,")]");
    if((I.c - F.c + 2*Pc)/Sc != O.c)
        err("[(I.c=",I.c,")-(F.c=",F.c,")+2*(Pc=",Pc,")]/(Sc=",Sc,")=[",(I.c-F.c+2*Pc)/Sc,"!=(O.c=",O.c,")]");
    if(I.ch != F.ch)
        err("(I.ch=",I.ch,")!=(F.ch=",F.ch,')');
    if(I.ch != O.ch)
        err("(I.ch=",I.ch,")!=(O.ch=",O.ch,')');
    if(F.ch != O.ch)
        err("(F.ch=",F.ch,")!=(O.ch=",O.ch,')');
    #endif

    unsigned constants[N_CONVOLVE_CONSTANTS]
    {
        Ir,Ic,
        Fr,Fc,
        Or,Oc,Ch,
        Sr,Sc,
        Pr,Pc,
        Ir*Ic,
        Fr*Fc,
        Or*Oc
    };
    /*
    constants[_iIr_] = Ir;
    constants[_iIc_] = Ic;
    constants[_iFr_] = Fr;
    constants[_iFc_] = Fc;
    constants[_iOr_] = Or;
    constants[_iOc_] = Oc;
    constants[_iCh_] = Ch;
    constants[_iSr_] = Sr;
    constants[_iSc_] = Sc;
    constants[_iPr_] = Pr;
    constants[_iPc_] = Pc;
    constants[_iIrIc_] = Ir*Ic;
    constants[_iFrFc_] = Fr*Fc;
    constants[_iOrOc_] = Or*Oc;
    */
    GPU::check(
        cudaMemcpyToSymbolAsync(
            CONVOLVE_CONSTANTS,
            constants,
            N_CONVOLVE_CONSTANTS*sizeof(unsigned),
            0,
            cudaMemcpyHostToDevice,
            stream
        )
    );
}
__inline__ __device__ double warpReduceSum(double v)
{
    double res = v;
    #pragma unroll
    for(unsigned i = warpSize / 2;i;i >>= 1)
        res += __shfl_down_sync(~0,res,i);
    return res;
}
__inline__ __device__ double blockReduceSum(double v)
{
    extern __shared__ double result[];
    const unsigned idx = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;

    // Reduce in every warp
    double warp = warpReduceSum(v);
    if(!(idx%warpSize))
        result[idx/warpSize] = warp;
    __syncthreads();

    // Since max threads per is coincidentally (warpSize*warpSize), the
    if(idx < warpSize)
    {
        const unsigned blk = (blockDim.x*blockDim.y*blockDim.z+warpSize-1)/warpSize;
        double block = warpReduceSum(idx < blk? warp:0);
        __syncwarp();
        if(!idx) result[0] = block;
    }
    __syncthreads();
    double out = result[0];
    __syncthreads();
    return out;
}
__global__ void convolveKernel(double *I,double *F,double *O,double b)
{
    using convolve::CONVOLVE_CONSTANTS;
    // Compute the local result.
    const unsigned Or = blockIdx.y,
                  SOr = _Sr_*Or,
               dPrSOr = _Pr_-SOr,
                  fr0 = _Pr_ > SOr? dPrSOr : 0U,
                  fr1 = umax(_Fr_,_Ir_+dPrSOr),
                   fr = threadIdx.y;
    double localRes = 0;
    if(fr0 <= fr && fr < fr1)
    {
        const unsigned Oc = blockIdx.x,
                      SOc = _Sc_*Oc,
                   dPcSOc = _Pc_-SOc,
                      fc0 = _Pc_ > SOc? dPcSOc : 0U,
                      fc1 = umin(_Fc_,_Ic_+dPcSOc),
                       fc = threadIdx.x;
        if(fc0 <= fc && fc < fc1)
        {
            const unsigned ch = blockIdx.z,
                          ich = _IrIc_ * ch,
                          fch = _FrFc_ * ch;
            localRes = I[ich + _Ic_ * (fr - dPrSOr) + fc - dPcSOc]
                     * F[fch + _Fc_ * fr + fc];
        }
    }
    __syncthreads();

    // Reduce via shuffling.
    double out = blockReduceSum(localRes);
    if(!(threadIdx.x|threadIdx.y))
        O[_OrOc_*blockIdx.z+_Oc_*blockIdx.y+blockIdx.x] = fmax(0.,out+b);
    __syncthreads();
}
void convolve::convolveGPU(GPUMatrix &I,GPUMatrix &F,GPUMatrix &O,
                           unsigned Sr,unsigned Sc,
                           unsigned Pr,unsigned Pc,
                           double bias)
{
    #ifdef DEBUG
    if(I.stream != F.stream || I.stream != O.stream || F.stream != O.stream)
        err("Stream mismatches");
    if(F.r*F.c > GPU::properties.maxThreadsPerBlock)
        err("Filter size too large (",F.r,'*',F.c,'>',GPU::properties.maxThreadsPerBlock,").");
    #endif // DEBUG

    // Init
    dim3 block(F.c,F.r);
    dim3 grid(O.c,O.r,O.ch);

    // Send input data
    I.transferH2D();
    F.transferH2D();

    // Run kernel
    convolveKernel CONFIG4(
        grid,block,
        (block.y*block.x*sizeof(double)+(unsigned)GPU::properties.warpSize-1U)/
                      (unsigned)GPU::properties.warpSize,
        O.stream
    )(I.device,F.device,O.device,bias);

    // Transfer to host
    O.transferD2H();
    GPU::sync(O.stream);
}

unsigned computePadding(unsigned Os,unsigned Is,unsigned Fs,unsigned Ss) {return (Os*Ss+Fs-Is)/2;}
unsigned computeStep(unsigned Os,unsigned Is,unsigned Fs,unsigned Ps) {return (2*Ps+Is-Fs)/Os;}