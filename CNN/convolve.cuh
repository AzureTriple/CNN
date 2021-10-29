#pragma once

#include <cuda_runtime.h>
#include "matrix.h"
#include "gpu_matrix.cuh"

namespace convolve
{
    void convolveCPU(Matrix &I,Matrix &F,Matrix &O,
                     unsigned Sr,unsigned Sc,
                     unsigned Pr,unsigned Pc,
                     double bias);
    void convolveOMP(Matrix &I,Matrix &F,Matrix &O,
                     unsigned Sr,unsigned Sc,
                     unsigned Pr,unsigned Pc,
                     double bias);

    #define N_CONVOLVE_CONSTANTS 14
    __constant__ unsigned CONVOLVE_CONSTANTS[N_CONVOLVE_CONSTANTS];

    #define _iIr_ 0
    #define _iIc_ 1
    #define _iFr_ 2
    #define _iFc_ 3
    #define _iOr_ 4
    #define _iOc_ 5
    #define _iCh_ 6
    #define _iSr_ 7
    #define _iSc_ 8
    #define _iPr_ 9
    #define _iPc_ 10
    #define _iIrIc_ 11
    #define _iFrFc_ 12
    #define _iOrOc_ 13

    #define _Ir_ CONVOLVE_CONSTANTS[_iIr_]
    #define _Ic_ CONVOLVE_CONSTANTS[_iIc_]
    #define _Fr_ CONVOLVE_CONSTANTS[_iFr_]
    #define _Fc_ CONVOLVE_CONSTANTS[_iFc_]
    #define _Or_ CONVOLVE_CONSTANTS[_iOr_]
    #define _Oc_ CONVOLVE_CONSTANTS[_iOc_]
    #define _Ch_ CONVOLVE_CONSTANTS[_iCh_]
    #define _Sr_ CONVOLVE_CONSTANTS[_iSr_]
    #define _Sc_ CONVOLVE_CONSTANTS[_iSc_]
    #define _Pr_ CONVOLVE_CONSTANTS[_iPr_]
    #define _Pc_ CONVOLVE_CONSTANTS[_iPc_]
    #define _IrIc_ CONVOLVE_CONSTANTS[_iIrIc_]
    #define _FrFc_ CONVOLVE_CONSTANTS[_iFrFc_]
    #define _OrOc_ CONVOLVE_CONSTANTS[_iOrOc_]

    /*
    Puts constants used in the GPU convolution function into
    constant device memory.

    I_,F_,O_: Input, Filter, and Output
    S_,P_   : Step size and Padding size
    _r,_c   : # rows and # columns
    Ch      : # Channels (i.e. depth)
    */
    void convolveConstants(unsigned Ir,unsigned Ic,
                           unsigned Fr,unsigned Fc,
                           unsigned Or,unsigned Oc,
                           unsigned Ch,unsigned Sr,
                           unsigned Sc,unsigned Pr,
                           unsigned Pc,cudaStream_t stream);
    
    void convolveGPU(GPUMatrix &I,GPUMatrix &F,GPUMatrix &O,
                     unsigned Sr,unsigned Sc,
                     unsigned Pr,unsigned Pc,
                     double bias);
}