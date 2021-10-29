//#define DEBUG

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "convolve.cuh"
#include "matrix.h"
#include <iostream>
#include <ctime>
#include <omp.h>
#include <cuda_runtime.h>

#ifdef DEBUG
#define Ir 5U
#define Ic 8U

#define Fr 4U
#define Fc 5U

#define Or 6U
#define Oc 4U

#define Sr 1U
#define Sc 3U

#define Pr 2U
#define Pc 3U

#define Ch 1U
#else
#define Ir (1U<<11)
#define Ic (1U<<11)

#define Fr (1U<<8)
#define Fc (1U<<8)

#define Or (1U<<11)
#define Oc (1U<<11)

#define Sr 1U
#define Sc 1U

#define Pr ((Or*Sr+Fr-Ir)>>1)
#define Pc ((Oc*Sc+Fc-Ic)>>1)

#define Ch 3U

#include <random>

double next() {
    static thread_local std::mt19937 g(clock()+omp_get_thread_num());
    return std::uniform_real_distribution<double>()(g);
}
#endif

template<typename _M>
void p(const char *name,_M &M)
{
    printf("%s:\n",name);
    for(unsigned i = 0;i < M.ch;++i)
    {
        for(unsigned j = 0;j < M.r;++j)
        {
            for(unsigned k = 0;k < M.c;++k)
                printf("% 5d ",(unsigned)M(i,j,k));
            printf("\n");
        }
        printf("\n");
    }
}
void init(Matrix &out)
{
    #pragma omp parallel for
    for(long x = 0;x < (long)(out.ch*out.r*out.c);++x)
        out.data[x] = 
            #ifdef DEBUG
            (double)((long long)x - ((long long)out.ch*out.r*out.c+1)/2);
            #else
            next();
            #endif // DEBUG
}
void zero(Matrix &out)
{
    #pragma omp parallel for
    for(long x = 0;x < (long)(out.ch*out.r*out.c);++x)
        out.data[x] = 0;
}
template<typename Func,typename _M>
clock_t duration(const char *name,_M &I,_M &F,_M &O,Func func)
{
    #ifdef DEBUG
    p("I",I);
    p("F",F);
    printf("before:\n");
    p("O",O);
    #endif
    clock_t t0 = clock();
    func(I,F,O,Sr,Sc,Pr,Pc,0.);
    clock_t t1 = clock();
    #ifdef DEBUG
    printf("after:\n");
    p("O",O);
    #endif
    printf("%s: %f\n",name,((long long)t1-t0)/(double)CLOCKS_PER_SEC);
    return t1-t0;
}

int main()
{
    Matrix I(Ch,Ir,Ic),
           F(Ch,Fr,Fc),
           O(Ch,Or,Oc);
    init(I);
    init(F);
    zero(O);

    // I disabled the single-threaded CPU test because it takes too
    // long to run.
    
    //clock_t CPU = duration("CPU",I,F,O,convolve::convolveCPU);
    //zero(O);
    clock_t OMP = duration("OMP",I,F,O,convolve::convolveOMP);
    zero(O);
    
    clock_t GPU;
    cudaStream_t stream = GPU::createStream();
    {
        GPUMatrix gI(I,stream,cudaHostAllocWriteCombined),
                  gF(F,stream,cudaHostAllocWriteCombined),
                  gO(O,stream);
        convolve::convolveConstants(Ir,Ic,Fr,Fc,
                                    Or,Oc,Ch,
                                    Sr,Sc,Pr,Pc,
                                    stream);
        GPU = duration("GPU",gI,gF,gO,convolve::convolveGPU);
    }
    GPU::destroyStream(stream);

    //printf("CPU-OMP=%ld\n",CPU-OMP);
    //printf("CPU-GPU=%ld\n",CPU-GPU);
    printf("OMP-GPU=%ld\n",OMP-GPU);

    GPU::sync();
}