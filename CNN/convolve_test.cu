#include "convolve_test.cuh"
#include <iostream>

constexpr const unsigned Ir = 4,Ic = 3,Ch = 3,
                         Fr = 2,Fc = 3,nF = 2,
                         Or = 5,Oc = 3,
                         Pr = 1,Pc = 2,
                         Sr = 1,Sc = 2;
constexpr const double LR = 1;

constexpr const double INI_I[Ir*Ic*Ch] {-18,-6, 6,  -17,-5, 7,  -16,-4, 8,
                                        -15,-3, 9,  -14,-2,10,  -13,-1,11,
                                        -12, 0,12,  -11, 1,13,  -10, 2,14,
                                        - 9, 3,15,  - 8, 4,16,  - 7, 5,17};

constexpr const double INI_F[nF*Fr*Fc*Ch] {- 9,-3, 3,  - 8,-2, 4,  - 7,-1, 5,
                                           - 6, 0, 6,  - 5, 1, 7,  - 4, 2, 8,

                                           -18,-6, 6,  -16,-4, 8,  -14,-2,10,
                                           -12, 0,12,  -10, 2,14,  - 8, 4,16};

constexpr const double INI_B[nF] {3,4};

constexpr const double INI_dO[Or*Oc*nF] {-6,-12,  -5,-10,  -4,- 8,
                                         -3,- 6,  -2,- 4,  -1,- 2,
                                          0,  0,   1,  2,   2,  4,
                                          3,  6,   4,  8,   5, 10,
                                          6, 12,   7, 14,   8, 16};


constexpr const double EXP_O[Or*Oc*nF] {111,220,  396, 790,  147,292,
                                        291,580,  951,1900,  327,652,
                                        300,598,  924,1846,  300,598,
                                        309,616,  897,1792,  273,544,
                                        138,274,  369, 736,  102,202};

constexpr const double EXP_F[nF*Fr*Fc*Ch] { 197,- 85,-367,   72,-42,-156,   99,- 87,-273,
                                           -100,- 94,- 88, - 75,-45,- 15, -198,- 96,   6,
                                            
                                            394,-170,-734,  144,-84,-312,  198,-174,-546,
                                           -200,-188,-176, -150,-90,- 30, -396,-192,  12};

constexpr const double EXP_B[nF] {-12,-26};

constexpr const double EXP_dI[Ir*Ic*Ch] { 465,- 15,-495,   205,- 5,-215,   335,- 25,-385,
                                           75,- 45,-165,    10,-20,- 50,  - 55,- 55,- 55,
                                         -315,- 75, 165,  -185,-35, 115,  -445,- 85, 275,
                                         -705,-105, 495,  -380,-50, 280,  -835,-115, 605};

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

#define CMP4D(name,W,X,Y,Z,exp,res) \
{\
    printf(name ":\n");\
    hasErr = 0;\
    for(unsigned w = 0;w < W;++w)\
    {\
        for(unsigned x = 0;x < X;++x)\
        {\
            for(unsigned y = 0;y < Y;++y)\
            {\
                for(unsigned z = 0;z < Z;++z)\
                {\
                    unsigned idx = w*X*Y*Z+x*Y*Z+y*Z+z;\
                    if(exp[idx] != res[idx])\
                    {\
                        hasErr = 1;\
                        printf(\
                            "\t<%u,%u,%u,%u> %f != %f\n",\
                            w,x,y,z,exp[idx],res[idx]\
                        );\
                    }\
                }\
            }\
        }\
    }\
    if(!hasErr) printf("\tNo Errors.\n");\
}

#include "convolve.cuh"
void convolve_test::testSTC()
{
    bool hasErr;
    double *I,*dI;
    conv::STCLayer<double> layer(Or,Oc,nF,
                                 Fr,Fc,Ch,
                                 Pr,Pc,
                                 Sr,Sc,LR,
                                 &I,&dI);
    
    I = new double[Ir*Ic*Ch];
    CPY(   Ir*Ic*Ch,      I,INI_I);
    CPY(nF*Fr*Fc*Ch,layer.F,INI_F);
    CPY(nF         ,layer.B,INI_B);
    layer.forward();
    CMP3D("O",Or,Oc,nF,EXP_O,layer.O);
    if(hasErr)
    {
        delete[] I;
        return;
    }

    layer.dO = new double[Or*Oc*nF];
    CPY(Or*Oc*nF,layer.dO,INI_dO);
    layer.backward();
    CMP4D("F",nF,Fr,Fc,Ch,EXP_F,layer.F);
    CMP1D("B",nF,EXP_B,layer.B);
    CMP3D("dI",Ir,Ic,Ch,EXP_dI,dI);

    delete[] I,dI;
}
void convolve_test::testOMP()
{
    bool hasErr;
    double *I,*dI;
    conv::OMPLayer<double> layer(Or,Oc,nF,
                                 Fr,Fc,Ch,
                                 Pr,Pc,
                                 Sr,Sc,LR,
                                 &I,&dI);
    
    I = new double[Ir*Ic*Ch];
    CPY(   Ir*Ic*Ch,      I,INI_I);
    CPY(nF*Fr*Fc*Ch,layer.F,INI_F);
    CPY(nF         ,layer.B,INI_B);
    layer.forward();
    CMP3D("O",Or,Oc,nF,EXP_O,layer.O);
    if(hasErr)
    {
        delete[] I;
        return;
    }

    layer.dO = new double[Or*Oc*nF];
    CPY(Or*Oc*nF,layer.dO,INI_dO);
    layer.backward();
    CMP4D("F",nF,Fr,Fc,Ch,EXP_F,layer.F);
    CMP1D("B",nF,EXP_B,layer.B);
    CMP3D("dI",Ir,Ic,Ch,EXP_dI,dI);

    delete[] I,dI;
}
void convolve_test::testGPU()
{
    cudaStream_t stream = GPU::createStream();

    bool hasErr;
    double *I,*dI,*d_I,*h_dI;
    conv::GPULayer<double> layer(Or,Oc,nF,
                                 Fr,Fc,Ch,
                                 Pr,Pc,
                                 Sr,Sc,LR,
                                 &I,&dI,
                                 stream,&d_I);

    GPU::allocHostPinned(&I,Ir*Ic*Ch);
    CPY(   Ir*Ic*Ch,      I,INI_I);
    CPY(nF*Fr*Fc*Ch,layer.F,INI_F);
    CPY(nF         ,layer.B,INI_B);
    GPU::allocTransfer(I,layer.d_I,Ir*Ic*Ch,stream);
    layer.forward();
    GPU::destroyDeviceMem(layer.d_O,stream);
    layer.d_O = nullptr;
    GPU::sync(stream);
    CMP3D("O",Or,Oc,nF,EXP_O,layer.O);
    if(hasErr)
    {
        GPU::destroyHostPinned(I);
        GPU::sync(stream);
        GPU::destroyStream(stream);
        return;
    }

    GPU::allocTransfer((double*)INI_dO,&layer.dO,Or*Oc*nF,stream);
    layer.backward();
    GPU::allocHostPinned(&h_dI,Ir*Ic*Ch);
    GPU::destroyTransfer(*layer.dI,h_dI,Ir*Ic*Ch,stream);
    *layer.dI = nullptr;
    GPU::sync(stream);
    CMP4D("F",nF,Fr,Fc,Ch,EXP_F,layer.F);
    CMP1D("B",nF,EXP_B,layer.B);
    CMP3D("dI",Ir,Ic,Ch,EXP_dI,h_dI);

    GPU::destroyHostPinned(I);
    GPU::destroyHostPinned(h_dI);
    GPU::sync(stream);
    GPU::destroyStream(stream);
}


#include <map>
#include <set>
void p(unsigned ir,unsigned ic,unsigned ch,unsigned or,unsigned oc,unsigned f)
{
    printf("I<%u,%u,%u>dO<%u,%u,%u> ",ir,ic,ch,or,oc,f);
}
void t(unsigned f,unsigned fr,unsigned fc,unsigned ch)
{
    printf("F<%u,%u,%u,%u>:",f,fr,fc,ch);
}
void ln() {printf("\n");}
void tab() {printf("\n\t");}
struct _IO {
    unsigned ir,ic,ch,or,oc,f;

    void p() const {printf("I<%u,%u,%u>dO<%u,%u,%u>",ir,ic,ch,or,oc,f);}
};
struct _F {
    unsigned f,fr,fc,ch;

    void p() const {printf("F<%u,%u,%u,%u>",f,fr,fc,ch);}
};
struct _FC {
    bool operator()(const _F &a,const _F &b) const
    {
        return a.f  < b.f  || (a.f  == b.f  && (
               a.fr < b.fr || (a.fr == b.fr && (
               a.fc < b.fc || (a.fc == b.fc && (
               a.ch < b.ch
               ))))));
    }
};
struct _IOC {
    bool operator()(const _IO &a,const _IO &b) const
    {
        return a.ir < b.ir || (a.ir == b.ir && (
               a.ic < b.ic || (a.ic == b.ic && (
               a.ch < b.ch || (a.ch == b.ch && (
               a.or < b.or || (a.or == b.or && (
               a.oc < b.oc || (a.oc == b.oc && (
               a.f  < b.f
               ))))))))));
    }
};
using set = std::set<_IO,_IOC>;
using map = std::map<_F,set,_FC>;
void good_dF(map &m)
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
                const unsigned _och = _oc+f;
                               //_fch = Fr*Fc*Ch*f;
                if(EXP_O[_och])
                {
                    //const double &dLdO = INI_dO[_och] * LR;
                    for(unsigned fr = fr0;fr < fr1;++fr)
                    {
                        //const unsigned //_ir = Ic*Ch*(fr-dprsor),
                                       //_fr = _fch+Fc*Ch*fr;
                        for(unsigned fc = fc0;fc < fc1;++fc)
                        {
                            //const unsigned //_ic = _ir+Ch*(fc-dpcsoc),
                                           //_fc = _fr+Ch*fc;
                            for(unsigned ch = 0;ch < Ch;++ch)
                            {
                                //const unsigned //_ix = _ic+ch,
                                               //_fx = _fc+ch;
                                //dI[_ix] += dLdO * F[_fx];
                                //dF[_fx] += dLdO * I[_ix];
                                m[{f,fr,fc,ch}].insert({fr-dprsor,fc-dpcsoc,ch,or,oc,f});
                            }
                        }
                    }
                }
            }
        }
    }
}
void bad_dF(map &m)
{
    double *ndo = (double*)malloc(sizeof(double)*Or*Oc*nF);
    for(unsigned o = 0;o < Or*Oc*nF;++o)
        ndo[o] = EXP_O[o]? INI_dO[o]*LR:0;
    for(long long x = 0;x < (long long)nF*Fr*Fc*Ch;++x)
    {
        const unsigned f = (unsigned)(x/Ch/Fc/Fr),
                      fr = (unsigned)((x/Ch/Fc)%Fr),
                      fc = (unsigned)((x/Ch)%Fc),
                      ch = (unsigned)(x%Ch),
                   dprfr = Pr-fr,
                   dpcfc = Pc-fc,
                     or0 = Pr > fr? (dprfr+Sr-1)/Sr : 0,
                     oc0 = Pc > fc? (dpcfc+Sc-1)/Sc : 0,
                     or1 = std::min<unsigned>(Or,(Ir-1+dprfr)/Sr+1),
                     oc1 = std::min<unsigned>(Oc,(Ic-1+dpcfc)/Sc+1);
        set &s = m[{f,fr,fc,ch}];
        for(unsigned or = or0;or < or1;++or)
        {
            const unsigned //_or = Oc*nF*or+f,
                            ir = or*Sr-dprfr;
            for(unsigned oc = oc0;oc < oc1;++oc)
                //if(ndo[_or+nF*oc])
                    s.insert({ir,oc*Sc-dpcfc,ch,or,oc,f});
        }
    }
    free(ndo);
}
#include <algorithm>
using std::set_difference;
#include <iterator>
using std::inserter;
void convolve_test::dbg()
{
    map g,b;
    good_dF(g);
    bad_dF(b);
    for(auto f = g.begin();f != g.end();++f)
    {
        f->first.p();
        printf(":");
        set &gs = f->second,&bs = b[f->first],
             bg,gb;
        set_difference(
            gs.begin(),gs.end(),
            bs.begin(),bs.end(),
            inserter(gb,gb.begin()),
            _IOC{}
        );
        for(auto i = gb.begin();i != gb.end();++i)
        {
            tab();
            printf("-");
            i->p();
        }
        set_difference(
            bs.begin(),bs.end(),
            gs.begin(),gs.end(),
            inserter(bg,bg.begin()),
            _IOC{}
        );
        for(auto i = bg.begin();i != bg.end();++i)
        {
            tab();
            printf("+");
            i->p();
        }
        ln();ln();
        b.erase(f->first);
    }
    for(auto f = b.begin();f != b.end();++f)
    {
        f->first.p();
        set &bs = f->second;
        for(auto i = bs.begin();i != bs.end();++i)
        {
            tab();
            printf("+");
            i->p();
        }
    }
}