#include "pool_test.cuh"
#include "pool.cuh"

constexpr const unsigned Ir = 6,Ic = 6,Ich = 2,
                         Or = 2,Oc = 3,Och = 1,
                         Pr = 3,Pc = 2,Pch = 2;

constexpr const double INI_I[Ir*Ic*Ich] = { 0,35,   4,31,   8,27,  12,23,  16,19,   1,34,
                                           19,16,  20,15,  24,11,  28, 7,  21,14,   5,30,
                                           15,20,  31, 4,  32, 3,  33, 2,  25,10,   9,26,
                                           11,24,  27, 8,  35, 0,  34, 1,  29, 6,  13,22,
                                            7,28,  23,12,  30, 5,  26, 9,  22,13,  17,18,
                                            3,32,  18,17,  14,21,  10,25,   6,29,   2,33};
constexpr const double INI_dO[Or*Oc*Och] = {-3,-2,-1,
                                             0, 1, 2};

constexpr const double EXP_O[Or*Oc*Och] = {35,33,34,
                                           32,35,33};
constexpr const double EXP_dI[Ir*Ic*Ich] = {0,-3, 0,0, 0,0,  0,0, 0,0, 0,-1,
                                            0, 0, 0,0, 0,0,  0,0, 0,0, 0, 0,
                                            0, 0, 0,0, 0,0, -2,0, 0,0, 0, 0,
                                            0, 0, 0,0, 1,0,  0,0, 0,0, 0, 0,
                                            0, 0, 0,0, 0,0,  0,0, 0,0, 0, 0,
                                            0, 0, 0,0, 0,0,  0,0, 0,0, 0, 2};

#define CPY(size,dst,src) \
for(unsigned i = 0;i < size;++i)\
    dst[i] = src[i]

#define CMP1D(name,X,exp,res) \
{\
    printf(name ":\n");\
    hasErr = 0;\
    for(unsigned x = 0;x < X;++x)\
    {\
        if(exp[x] != res[x])\
        {\
            hasErr = 1;\
            printf(\
                "\t<%u> %f != %f\n",\
                x,exp[x],res[x]\
            );\
        }\
    }\
    if(!hasErr) printf("\tNo Errors.\n");\
}

#define CMP3D(name,X,Y,Z,exp,res) \
{\
    printf(name ":\n");\
    hasErr = 0;\
    for(unsigned x = 0;x < X;++x)\
    {\
        for(unsigned y = 0;y < Y;++y)\
        {\
            for(unsigned z = 0;z < Z;++z)\
            {\
                unsigned idx = x*Y*Z+y*Z+z;\
                if(exp[idx] != res[idx])\
                {\
                    hasErr = 1;\
                    printf(\
                        "\t<%u,%u,%u> %f != %f\n",\
                        x,y,z,exp[idx],res[idx]\
                    );\
                }\
            }\
        }\
    }\
    if(!hasErr) printf("\tNo Errors.\n");\
}

void pool_test::testSTC()
{
    bool hasErr;
    double *O = (double*)malloc(sizeof(double)*Or*Oc*Och),*dO;
    pool::STCLayer<double> layer(&O,&dO,Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch);
    CPY(Ir*Ic*Ich,layer.I,INI_I);

    layer.forward();
    CMP3D("O",Or,Oc,Och,EXP_O,O);
    free(O);
    if(hasErr) return;

    dO = (double*)malloc(sizeof(double)*Or*Oc*Och);
    CPY(Or*Oc*Och,dO,INI_dO);

    layer.backward();
    CMP3D("dI",Ir,Ic,Ich,EXP_dI,layer.dI);
}
void pool_test::testOMP()
{
    bool hasErr;
    double *O = (double*)malloc(sizeof(double)*Or*Oc*Och),*dO;
    pool::OMPLayer<double> layer(&O,&dO,Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch);
    CPY(Ir*Ic*Ich,layer.I,INI_I);

    layer.forward();
    CMP3D("O",Or,Oc,Och,EXP_O,O);
    free(O);
    if(hasErr) return;

    dO = (double*)malloc(sizeof(double)*Or*Oc*Och);
    CPY(Or*Oc*Och,dO,INI_dO);

    layer.backward();
    CMP3D("dI",Ir,Ic,Ich,EXP_dI,layer.dI);
}
void pool_test::testGPU()
{
    cudaStream_t stream = GPU::createStream();

    bool hasErr;
    double *O,*dO,*h_O,*h_dI;

    pool::GPULayer<double> layer(&O,&dO,&h_O,Ir,Ic,Ich,Or,Oc,Och,Pr,Pc,Pch);

    GPU::allocHostPinned(&h_O,Or*Oc*Och);
    GPU::allocDeviceMem(&layer.I,Ir*Ic*Ich,stream);
    GPU::transfer<double,cudaMemcpyHostToDevice>((double*)INI_I,layer.I,Ir*Ic*Ich,stream);
    layer.forward(stream);
    GPU::transfer<double,cudaMemcpyDeviceToHost>(O,h_O,Or*Oc*Och,stream);
    GPU::destroyDeviceMem(O,stream);
    GPU::sync(stream);
    CMP3D("O",Or,Oc,Och,EXP_O,h_O);
    GPU::destroyHostPinned(h_O);
    if(hasErr)
    {
        GPU::sync(stream);
        GPU::destroyStream(stream);
        GPU::sync();
        return;
    }
    
    GPU::allocHostPinned(&h_dI,Ir*Ic*Ich);
    GPU::allocDeviceMem(&dO,Or*Oc*Och,stream);
    GPU::transfer<double,cudaMemcpyHostToDevice>((double*)INI_dO,dO,Or*Oc*Och,stream);
    layer.backward(stream);
    GPU::transfer<double,cudaMemcpyDeviceToHost>(layer.dI,h_dI,Ir*Ic*Ich,stream);
    GPU::destroyDeviceMem(layer.dI,stream);
    GPU::sync(stream);
    CMP3D("dI",Ir,Ic,Ich,EXP_dI,h_dI);
    GPU::destroyHostPinned(h_dI);

    GPU::sync(stream);
    GPU::destroyStream(stream);
    GPU::sync();
}