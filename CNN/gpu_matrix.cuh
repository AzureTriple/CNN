#pragma once

#include <cstring>        // memcpy
#include <cuda_runtime.h>
#include "GPU.cuh"
#include "matrix.h"

struct GPUMatrix {
    double *host,*device;
    unsigned ch,r,c;
    cudaStream_t stream;

    GPUMatrix(unsigned ch,unsigned r,unsigned c,cudaStream_t stream,unsigned mode = cudaHostAllocDefault) 
        : ch(ch),r(r),c(c),stream(stream)
    {
        GPU::allocDeviceMem(&device,(size_t)ch*r*c,stream);
        GPU::allocHostPinned<double>(&host,(size_t)ch*r*c,mode);
    }
    GPUMatrix(const Matrix &m,cudaStream_t stream,unsigned mode = cudaHostAllocDefault)
        : GPUMatrix(m.ch,m.r,m.c,stream,mode)
    {
        memcpy((void*)host,(void*)m.data,(size_t)ch*r*c*sizeof(double));
    }
    ~GPUMatrix()
    {
        GPU::destroyHostPinned(host);
        GPU::destroyDeviceMem(device,stream);
    }

    void transferH2D()
    {
        GPU::transfer<double,cudaMemcpyHostToDevice>(host,device,(size_t)ch*r*c,stream);
    }
    void transferD2H()
    {
        GPU::transfer<double,cudaMemcpyDeviceToHost>(device,host,(size_t)ch*r*c,stream);
    }

    double& operator()(unsigned _ch,unsigned _r,unsigned _c){return host[r*c*_ch+c*_r+_c];}
};