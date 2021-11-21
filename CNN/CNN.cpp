#define DEBUG

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#ifdef DBG
#include "pool_test.cuh"
#include <iostream>
#include <ctime>

template<typename F>
void duration(char const* name,F f)
{
    clock_t t0 = clock();
    f();
    clock_t t1 = clock();
    printf("%s: %f\n\n",name,((long long)t1-t0)/(double)CLOCKS_PER_SEC);
}

int main()
{
    using namespace pool_test;
    try {
        duration("STC",testSTC);
        duration("OMP",testOMP);
        duration("GPU",testGPU);
        //dbg();
    } catch(std::exception e) {
        std::cerr << e.what();
        return 1;
    }
}
#else
#include "convolve.cuh"
#include "pool.cuh"
#include "fcl.cuh"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>

std::ifstream train_labels("C:/Users/prgmPC/Desktop/School/Y4/ee/CNN/train-labels.idx1-ubyte",std::ifstream::binary);
std::ifstream train_images("C:/Users/prgmPC/Desktop/School/Y4/ee/CNN/train-images.idx3-ubyte",std::ifstream::binary);
std::ifstream test_labels("C:/Users/prgmPC/Desktop/School/Y4/ee/CNN/t10k-images.idx1-ubyte",std::ifstream::binary);
std::ifstream test_images("C:/Users/prgmPC/Desktop/School/Y4/ee/CNN/t10k-images.idx3-ubyte",std::ifstream::binary);

template<typename T,unsigned off>
T meta(std::ifstream &f)
{
    char buf[sizeof(T)];
    f.seekg(off).read(buf,sizeof(T));
    for(unsigned i = 0;i < sizeof(T)/2;++i)
    {
        char x = buf[sizeof(T)-i-1];
        buf[sizeof(T)-i-1] = buf[i];
        buf[i] = x;
    }
    return *(T*)(void*)buf;
}

const unsigned nTrain = meta<unsigned,4>(train_images),
               nTest = meta<unsigned,4>(test_images);
const unsigned Ir = meta<unsigned,8>(train_images),
               Ic = meta<unsigned,12>(train_images),
               Is = Ir*Ic;
char *buffer = (char*)malloc(sizeof(char)*Ir*Ic);

unsigned char label(std::ifstream &f,unsigned id) {return (unsigned char)f.seekg((size_t)id+8u).get();}
void image(std::ifstream &f,double *img,unsigned id)
{
    f.seekg((size_t)id*Is+16u).read(buffer,Is);
    for(unsigned i = 0;i < Ir*Ic;++i)
        img[i] = buffer[i]/255.;
}

#include <random>
std::default_random_engine generator;
void initialize(double *W,unsigned size,unsigned nI,unsigned nO)
{
    double scale = sqrt(2./nI/nO);
    std::uniform_real_distribution<double> distribution(std::min(nI,nO),std::max(nI,nO));
    for(unsigned i = 0;i < size;++i)
        W[i] = distribution(generator)*scale;
}
void initialize(double *W,unsigned size,unsigned nI,unsigned nO,unsigned N)
{
    double scale = sqrt(2./N);
    std::uniform_real_distribution<double> distribution(std::min(nI,nO),std::max(nI,nO));
    for(unsigned i = 0;i < size;++i)
        W[i] = distribution(generator)*scale;
}

void shuffle(unsigned *order,unsigned size)
{
    while(--size > 1)
    {
        const unsigned x = rand() % size;
        const unsigned y = order[x];
        order[x] = order[size];
        order[size] = y;
    }
}
void softmax(double *O)
{
    double sumExp(0);
    for(unsigned i = 0;i < 10;++i)
        sumExp += O[i] = exp(O[i]);
    for(unsigned i = 0;i < 10;++i)
        O[i] /= sumExp;
}
void eval(unsigned expected,double *O,double *dO)
{
    double sumExp(0);
    for(unsigned i = 0;i < 10;++i)
        sumExp += O[i] = exp(O[i]);
    for(unsigned i = 0;i < 10;++i)
    {
        O[i] /= sumExp;
        dO[i] = O[i] * (!(i != expected) - O[i]);
    }
}

constexpr const unsigned N_EPOCHS = 1<<6,N_CONV = 4,N_FCL = 4,
                         FILTER_SIZE = 15,FCL_SIZE = 1024,N_CH = 4;
constexpr const double LR = 0.01;

#define MAKE(type,size,name) type *name = new type[size]

int runSTC(unsigned scale)
{
    if(!nTrain) return 1;
    scale += 2;
    double *dO;
    MAKE(double,10,O); //if(!O) return 1;
    MAKE(unsigned,nTrain,order); //if(!order) {free(O); return 1;}
    MAKE(convolve::STCLayer<double>*,N_CONV,conv); //if(!conv) {free(O); free(order); return 1;}
    MAKE(    pool::STCLayer<double>*,N_CONV,pool); //if(!pool) {free(O); free(order); free(conv); return 1;}
    MAKE(     fcl::STCLayer<double>*,N_FCL,fcl);   //if(!fcl ) {free(O); free(order); free(conv); free(pool); return 1;}
    for(unsigned i = 0;i < nTrain;++i) order[i] = i;

    fcl[N_FCL-1] = new fcl::STCLayer<double>(FCL_SIZE,10,&O,&dO,LR);
    initialize(fcl[N_FCL-1]->W,FCL_SIZE*10,FCL_SIZE,10);
    for(unsigned f = N_FCL-1;f;--f)
    {
        fcl[f-1] = new fcl::STCLayer<double>(Ir*Ic,FCL_SIZE,&fcl[f]->I,&fcl[f]->dI,LR);
        initialize(fcl[f]->W,Ir*Ic*FCL_SIZE,Ir*Ic,FCL_SIZE);
    }

    const unsigned Pr = (FILTER_SIZE-1)/2,Pc = (FILTER_SIZE-1)/2;
    pool[N_CONV-1] = new pool::STCLayer<double>(&fcl[0]->I,&fcl[0]->dI,
                                                Ir,Ic,N_CH*scale,
                                                Ir,Ic,1,
                                                1,1,N_CH*scale);
    for(unsigned c = N_CONV-1;c;)
    {
        conv[c] = new convolve::STCLayer<double>(Ir,Ic,
                                                 FILTER_SIZE,FILTER_SIZE,N_CH*scale,
                                                 Ir,Ic,N_CH,
                                                 Pr,Pc,1,1,
                                                 &pool[c]->I,&pool[c]->dI,
                                                 LR);
        initialize(conv[c]->F,FILTER_SIZE*FILTER_SIZE*N_CH*scale,Ir*Ic*N_CH,Ir*Ic*N_CH*scale);
        --c;
        pool[c] = new pool::STCLayer<double>(&conv[c+1]->I,&conv[c+1]->dI,
                                             Ir,Ic,N_CH*scale,
                                             Ir,Ic,N_CH,
                                             1,1,scale);
    }
    conv[0] = new convolve::STCLayer<double>(Ir,Ic,
                                             FILTER_SIZE,FILTER_SIZE,N_CH*scale,
                                             Ir,Ic,1,
                                             Pr,Pc,1,1,
                                             &pool[0]->I,&pool[0]->dI,
                                             LR);
    initialize(conv[0]->F,FILTER_SIZE*FILTER_SIZE*N_CH*scale,Ir*Ic,Ir*Ic*N_CH*scale);

    for(unsigned epoch = 0;epoch < N_EPOCHS;++epoch)
    {
        shuffle(order,nTrain);
        for(unsigned i = 0;i < nTrain;++i)
        {
            image(train_images,conv[0]->I,order[i]);
            for(unsigned l = 0;l < N_CONV;++l)
            {
                conv[l]->forward();
                pool[l]->forward();
            }
            for(unsigned l = 0;l < N_FCL;++l) fcl[l]->forward();
            dO = (double*)malloc(sizeof(double)*10);
            eval(label(train_labels,order[i]),O,dO);
            for(unsigned l = N_FCL;l;)
                fcl[--l]->backward();
            for(unsigned l = N_CONV;l;)
            {
                pool[--l]->backward();
                conv[l]->backward();
            }
        }
        unsigned correct = 0;
        for(unsigned i = 0;i < nTest;++i)
        {
            image(test_images,conv[0]->I,i);
            for(unsigned l = 0;l < N_CONV;++l)
            {
                conv[l]->forward();
                pool[l]->forward();
            }
            for(unsigned l = 0;l < N_FCL;++l) fcl[l]->forward();
            softmax(O);
            double max = O[0];
            unsigned guess = 0;
            for(unsigned t = 1;t < 10;++t)
            {
                if(max < O[t])
                {
                    max = O[t];
                    guess = t;
                }
            }
            unsigned expected = label(test_labels,i);
            if(guess == expected) ++correct;
        }
        printf("epoch %2u: %5u/%5u = %3.2f%%\n",epoch,correct,nTest,(double)correct/nTest);
    }
    for(unsigned l = 0;l < N_CONV;++l)
        delete conv[l],pool[l];
    for(unsigned l = 0;l < N_FCL;++l)
        delete fcl[l];
    delete[] conv,pool,fcl,O;
    //free(conv);
    //free(pool);
    //free(fcl);
    //free(O);
    return 0;
}

int main()
{
    srand((int)time(NULL));
    runSTC(0);
}
#endif