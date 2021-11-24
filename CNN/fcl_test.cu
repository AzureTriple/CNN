#include "fcl_test.cuh"

constexpr const unsigned Is = 20,Os = 4;

constexpr const double INI_I[Is] =    {-10,- 9,- 8,- 7,- 6,- 5,- 4,- 3,- 2,- 1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
constexpr const double INI_B[Is] =    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,-10,- 9,- 8,- 7,- 6,- 5,- 4,- 3,- 2,- 1};
constexpr const double INI_W[Is*Os] = {-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29,-28,-27,-26,-25,-24,-23,-22,-21,
                                        19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
                                       -20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,- 9,- 8,- 7,- 6,- 5,- 4,- 3,- 2,- 1,
                                        39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20};
constexpr const double INI_dO[Os] = {-2,-1,0,1};

constexpr const double EXP_O[Os] = {-1080,240,-280,1040};
constexpr const double EXP_dI[Is] =   {  0,  0,  0,  0,  0,  0, 88, 86, 84, 82,  0,  0,  0,  0,  0,  0, 68, 66, 64, 62};
constexpr const double EXP_B[Is] =    {  0,  1,  2,  3,  4,  5,-82,-79,-76,-73,-10,- 9,- 8,- 7,- 6,- 5,-72,-69,-66,-63};
constexpr const double EXP_W[Is*Os] = {-40,-39,-38,-37,-36,-35,-30,-25,-20,-15,-30,-29,-28,-27,-26,-25,-20,-15,-10,- 5,
                                        19, 18, 17, 16, 15, 14, 15, 16, 17, 18,  9,  8,  7,  6,  5,  4,  5,  6,  7,  8,
                                       -20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,- 9,- 8,- 7,- 6,- 5,- 4,- 3,- 2,- 1,
                                        39, 38, 37, 36, 35, 34, 31, 28, 25, 22, 29, 28, 27, 26, 25, 24, 21, 18, 15, 12};

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
                "\t<%2u> % 4.0f != % 4.0f\n",\
                x,exp[x],res[x]\
            );\
        }\
    }\
    if(!hasErr) printf("\tNo Errors.\n");\
}

#define CMP2D(name,X,Y,exp,res) \
{\
    printf(name ":\n");\
    hasErr = 0;\
    for(unsigned x = 0;x < X;++x)\
    {\
        for(unsigned y = 0;y < Y;++y)\
        {\
            unsigned idx = x*Y+y;\
            if(exp[idx] != res[idx])\
            {\
                hasErr = 1;\
                printf(\
                    "\t<%1u,%1u> % 4.0f != % 4.0f\n",\
                    x,y,exp[idx],res[idx]\
                );\
            }\
        }\
    }\
    if(!hasErr) printf("\tNo Errors.\n");\
}

#include "fcl.cuh"

void fcl_test::testSTC()
{
    bool hasErr;
    double *I,*dI;
    fcl::STCLayer<double> layer(Is,Os,&I,&dI,1);

    I = new double[Is];
    CPY(Is   ,      I,INI_I);
    CPY(Is   ,layer.B,INI_B);
    CPY(Is*Os,layer.W,INI_W);
    layer.forward();
    CMP1D("O",Os,EXP_O,layer.O);
    if(hasErr)
    {
        delete[] I;
        return;
    }

    layer.dO = new double[Os];
    CPY(Os,layer.dO,INI_dO);
    layer.backward();
    CMP1D("dI",Is,EXP_dI,dI);
    CMP1D("B",Is,EXP_B,layer.B);
    CMP2D("W",Is,Os,EXP_W,layer.W);

    delete[] I,dI;
}
void fcl_test::testOMP()
{
    bool hasErr;
    double *I,*dI;
    fcl::OMPLayer<double> layer(Is,Os,&I,&dI,1);

    I = new double[Is];
    CPY(Is   ,      I,INI_I);
    CPY(Is   ,layer.B,INI_B);
    CPY(Is*Os,layer.W,INI_W);
    layer.forward();
    CMP1D("O",Os,EXP_O,layer.O);
    if(hasErr)
    {
        delete[] I;
        return;
    }

    layer.dO = new double[Os];
    CPY(Os,layer.dO,INI_dO);
    layer.backward();
    CMP1D("dI",Is,EXP_dI,dI);
    CMP1D("B",Is,EXP_B,layer.B);
    CMP2D("W",Is,Os,EXP_W,layer.W);
    
    delete[] I,dI;
}
void fcl_test::testGPU()
{
    cudaStream_t stream = GPU::createStream();
    bool hasErr;
    double *I,*dI,*d_I,*h_dI;
    fcl::GPULayer<double> layer(Is,Os,(double**)&INI_I,&dI,1,
                                &d_I,stream);

    GPU::allocHostPinned(&I,Is);
    CPY(Is   ,      I,INI_I);
    CPY(Is   ,layer.B,INI_B);
    CPY(Is*Os,layer.W,INI_W);
    GPU::allocTransfer(I,layer.d_I,Is,stream);
    layer.forward();
    GPU::destroyDeviceMem(layer.d_O,stream);
    layer.d_O = nullptr;
    GPU::sync(stream);
    CMP1D("O",Os,EXP_O,layer.O);
    if(hasErr)
    {
        GPU::destroyHostPinned(I);
        GPU::sync(stream);
        GPU::destroyStream(stream);
        return;
    }

    GPU::allocTransfer((double*)INI_dO,&layer.dO,Os,stream);
    layer.backward();
    GPU::allocHostPinned(&h_dI,Is);
    GPU::destroyTransfer(*layer.dI,h_dI,Is,stream);
    *layer.dI = nullptr;
    GPU::sync(stream);
    CMP1D("dI",Is,EXP_dI,h_dI);
    CMP1D("B",Is,EXP_B,layer.B);
    CMP2D("W",Is,Os,EXP_W,layer.W);

    GPU::destroyHostPinned(I);
    GPU::destroyHostPinned(h_dI);
    GPU::sync(stream);
    GPU::destroyStream(stream);
}

void fcl_test::dbg()
{
    bool err = 0;
    constexpr const double exp[Is] = {0,0,0,0,0,0,2,4,6,8,0,0,0,0,0,0,2,4,6,8};
    for(unsigned i = 0;i < Is;++i)
    {
        const double o = std::max<double>(double(0),INI_I[i]+INI_B[i]);
        if(o != exp[i])
        {
            printf("IB<%2u> %f != %f\n",i,exp[i],o);
            err = 1;
        }
    }
    if(err) return;
    for(unsigned o = 0;o < Os;++o)
    {
        double v(0);
        const unsigned wo = Is*o;
        for(unsigned i = 0;i < Is;++i)
            v += exp[i] * INI_W[wo+i];
        if(v != EXP_O[o])
        {
            printf("O<%2u> %f != %f\n",o,EXP_O[o],v);
            for(unsigned i = 0;i < Is;++i)
                printf("\tI<%2u>*W<%2u,%2u|%2u>=% 3.0f*% 3.0f=% 4.0f\n",i,o,i,wo+i,exp[i],INI_W[wo+i],exp[i]*INI_W[wo+i]);
        }
    }
}