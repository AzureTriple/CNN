#include "mnist.cuh"
#include "convolve.cuh"
#include "pool.cuh"
#include "fcl.cuh"
#include "network.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>

std::ifstream train_labels,train_images,test_labels,test_images;

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
    return *(T *)(void *)buf;
}

unsigned nTrain,nTest,Ir,Ic,Is,
         N_EPOCHS,TRAIN_PER_EPOCH,TEST_PER_EPOCH;
char *buffer;
double LR;
constexpr const unsigned N_CONV = 4,N_FCL = 4,N_CH = 4,
                         FILTER_SIZE = 5,FCL_SIZE = 1024;

int mnist::init(const char *train_img,const char *train_lbl,
                const char *test_img,const char *test_lbl,
                unsigned epochs,unsigned train_count,
                unsigned test_count,double lr)
{
    train_images = std::ifstream(train_img,std::ifstream::binary);
    train_labels = std::ifstream(train_lbl,std::ifstream::binary);
    test_images = std::ifstream(test_img,std::ifstream::binary);
    test_labels = std::ifstream(test_lbl,std::ifstream::binary);
    if(!(train_images || train_labels || test_images || test_labels))
    {
        mnist::close();
        return 1;
    }

    nTrain = meta<unsigned,4>(train_images);
    nTest = meta<unsigned,4>(test_images);
    Ir = meta<unsigned,8>(train_images);
    Ic = meta<unsigned,12>(train_images);
    Is = Ir * Ic;
    if(!(train_images || train_labels || test_images || test_labels))
    {
        mnist::close();
        return 1;
    }

    buffer = new char[Is];
    N_EPOCHS = epochs;
    TRAIN_PER_EPOCH = train_count;
    TEST_PER_EPOCH = test_count;
    LR = lr;
    srand((int)time(NULL));
    printf("epochs=%u,n_train=%u,n_test=%u\n",N_EPOCHS,TRAIN_PER_EPOCH,TEST_PER_EPOCH);
    return 0;
}
void mnist::close()
{
    if(train_images) train_images.close();
    if(train_labels) train_labels.close();
    if(test_images) test_images.close();
    if(test_labels) test_labels.close();
}

unsigned char label(std::ifstream &f,unsigned id) {return (unsigned char)f.seekg((size_t)id+8u).get();}
void image(std::ifstream &f,double *img,unsigned id)
{
    f.seekg((size_t)id*Is+16u).read(buffer,Is);
    for(unsigned i = 0;i < Is;++i)
        img[i] = (unsigned char)buffer[i]/255.;
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

const double EXP_MIN = std::log(std::nextafter(0.,1.));
void softmax(double *O)
{
    double max = O[0];
    for(unsigned o = 1;o < 10;++o)
        if(max < O[o])
            max = O[o];
    double s = 0;
    for(unsigned o = 0;o < 10;++o)
        s += O[o] = std::exp(std::max<double>(O[o] - max,EXP_MIN));
    for(unsigned o = 0;o < 10;++o)
        O[o] /= s;
}
void eval(unsigned expected,double *O,double *dO)
{
    double max = O[0];
    for(unsigned o = 1;o < 10;++o)
        if(max < O[o])
            max = O[o];
    double s = 0;
    double nO[10]{};
    for(unsigned o = 0;o < 10;++o)
        s += nO[o] = std::exp(std::max<double>(O[o] - max,EXP_MIN));
    for(unsigned o = 0;o < 10;++o)
    {
        nO[o] /= s;
        //dO[o] = nO[o]-!(o!=expected);
        dO[o] = nO[o]*nO[o]*(1-nO[o]);
    }
    dO[expected] += nO[expected]*(nO[expected]-1);
}

void dbg_img(double *I)
{
    double maxv(0),minv(1),avgv(0);
    constexpr const double threshold = 0.5;
    for(unsigned r = 0;r < Ir;++r)
    {
        for(unsigned c = 0;c < Ic;++c)
        {
            const double &v = I[Ic*r+c];
            printf("%c",v < threshold? ' ':'#');
            maxv = std::max(maxv,v);
            minv = std::min(minv,v);
            if(v != 0) avgv += v;
        }
        printf("\n");
    }
    printf("max: %f | min: %f | avg: %f\n",maxv,minv,avgv/Is);
}

unsigned res_in(double *I)
{
    printf("result:\n");
    unsigned img = rand() % nTest;
    image(test_images,I,img);
    dbg_img(I);
    return img;
}
void res_out(double *O,unsigned img)
{
    softmax(O);
    printf("out:[%f",O[0]);
    double max = O[0];
    unsigned midx = 0;
    for(unsigned j = 1;j < 10;++j)
    {
        printf(" %f",O[j]);
        if(max < O[j])
        {
            max = O[j];
            midx = j;
        }
    }
    printf("]\nlabel: %u\npredicted: %u\n",label(test_labels,img),midx);
}
void network_result(network::Network<double> &network,double *I,double *O)
{
    const unsigned img = res_in(I);
    network.forward();
    res_out(O,img);
}
void gpu_result(network::Network<double> &network,
                double *I,double **d_I,
                cudaStream_t stream,
                double *d_O,double *O)
{
    const unsigned img = res_in(I);
    GPU::allocTransfer(I,d_I,Is,stream);
    network.forward();
    GPU::destroyDeviceMem(d_O,stream);
    GPU::sync(stream);
    res_out(O,img);
}
void orders(unsigned **train,unsigned **test)
{
    *train = new unsigned[nTrain];
    for(unsigned i = 0;i < nTrain;++i)
        (*train)[i] = i;
    *test = new unsigned[nTest];
    for(unsigned i = 0;i < nTest;++i)
        (*test)[i] = i;
}
bool check(double *O,unsigned label)
{
    softmax(O);
    double max = O[0];
    unsigned midx = 0;
    for(unsigned j = 1;j < 10;++j)
    {
        if(max < O[j])
        {
            max = O[j];
            midx = j;
        }
    }
    return midx == label;
}
void test_results(unsigned epoch,unsigned correct)
{
    printf(
        "epoch %2u: %3u/%3u = % 6.2f%%\n",
        epoch,correct,TEST_PER_EPOCH,
        100.*correct/TEST_PER_EPOCH
    );
}

mnist::Record mnist::runSTC(unsigned scale)
{
    Record out{0,0,0,0,0,0,0,0};
    if(!nTrain) return out;
    scale += 2;
    constexpr const unsigned Fr = FILTER_SIZE,Fc = FILTER_SIZE,
                             Pr = (Fr-1)/2,Pc = (Fc-1)/2;
    double *I = new double[Is],*dI = nullptr,*O,**dO;
    network::Network<double> *network;
    clock_t a,b;
    a = std::clock();
    {
        unsigned l = 0;
        layer::Layer<double> **layers = new layer::Layer<double>*[N_CONV*2+N_FCL];
        layers[l] = new conv::STCLayer<double>(Ir,Ic,N_CH*scale,
                                               Fr,Fc,1,
                                               Pr,Pc,1,1,LR,
                                               &I,&dI);
        ((conv::STCLayer<double>*)layers[l])->init();
        while(l < 2*(N_CONV-1))
        {
            ++l;
            layers[l] = new pool::STCLayer<double>(Ir,Ic,N_CH,
                                                   1,1,scale,
                                                   &layers[l-1]->O,
                                                   &layers[l-1]->dO);
            ++l;
            layers[l] = new conv::STCLayer<double>(Ir,Ic,N_CH*scale,
                                                   Fr,Fc,N_CH,
                                                   Pr,Pc,1,1,LR,
                                                   &layers[l-1]->O,
                                                   &layers[l-1]->dO);
            ((conv::STCLayer<double>*)layers[l])->init();
        }
        ++l;
        layers[l] = new pool::STCLayer<double>(Ir,Ic,1,
                                               1,1,N_CH*scale,
                                               &layers[l-1]->O,
                                               &layers[l-1]->dO);
        ++l;
        layers[l] = new fcl::STCLayer<double>(Is,FCL_SIZE,
                                              &layers[l-1]->O,
                                              &layers[l-1]->dO,
                                              LR);
        ((fcl::STCLayer<double>*)layers[l])->init();
        for(unsigned f = 1;f < N_FCL-1;++f)
        {
            ++l;
            layers[l] = new fcl::STCLayer<double>(FCL_SIZE,FCL_SIZE,
                                                  &layers[l-1]->O,
                                                  &layers[l-1]->dO,
                                                  LR);
            ((fcl::STCLayer<double>*)layers[l])->init();
        }
        ++l;
        layers[l] = new fcl::STCLayer<double>(FCL_SIZE,10,
                                              &layers[l-1]->O,
                                              &layers[l-1]->dO,
                                              LR);
        ((fcl::STCLayer<double>*)layers[l])->init();
        dO = &layers[l]->dO;
        O = layers[l]->O;
        network = new network::Network<double>(layers,l+1);
    }
    b = std::clock();
    out.setup = (double)(b - a) / CLOCKS_PER_SEC;
    unsigned *order_train,*order_test;
    orders(&order_train,&order_test);
    bool first = true;
    clock_t maxF,minF,maxB,minB,diff;
    out.avgF = out.avgB = 0;
    for(unsigned epoch = 0;epoch < N_EPOCHS;++epoch)
    {
        shuffle(order_train,nTrain);
        for(unsigned i = 0;i < TRAIN_PER_EPOCH;++i)
        {
            image(train_images,I,order_train[i]);

            a = std::clock();
            network->forward();
            b = std::clock();
            diff = b - a;
            if(first) maxF = minF = diff;
            else
            {
                maxF = std::max(maxF,diff);
                minF = std::min(minF,diff);
            }
            out.avgF += (double)diff/TRAIN_PER_EPOCH/N_EPOCHS/CLOCKS_PER_SEC;

            eval(label(train_labels,order_train[i]),O,*dO = new double[10]);

            a = std::clock();
            network->backward();
            b = std::clock();
            diff = b - a;
            if(first) maxB = minB = diff;
            else
            {
                maxB = std::max(maxB,diff);
                minB = std::min(minB,diff);
            }
            out.avgB += (double)diff/TRAIN_PER_EPOCH/N_EPOCHS/CLOCKS_PER_SEC;

            delete[] dI;
            first = false;
        }
        shuffle(order_test,nTest);
        unsigned correct = 0;
        for(unsigned i = 0;i < TEST_PER_EPOCH;++i)
        {
            image(test_images,I,order_test[i]);
            network->forward();
            softmax(O);
            correct += check(O,label(test_labels,order_test[i]));
        }
        test_results(epoch,correct);
    }
    network_result(*network,I,O);
    out.highF = (double)maxF / CLOCKS_PER_SEC;
    out.highB = (double)maxB / CLOCKS_PER_SEC;
    out.lowF = (double)minF / CLOCKS_PER_SEC;
    out.lowB = (double)minB / CLOCKS_PER_SEC;
    a = std::clock();
    delete network;
    delete[] order_train,order_test,I;
    b = std::clock();
    out.tearDown = (double)(b - a) / CLOCKS_PER_SEC;
    return out;
}
mnist::Record mnist::runOMP(unsigned scale)
{
    Record out{0,0,0,0,0,0,0,0};
    if(!nTrain) return out;
    scale += 2;
    constexpr const unsigned Fr = FILTER_SIZE,Fc = FILTER_SIZE,
                             Pr = (Fr-1)/2,Pc = (Fc-1)/2;
    double *I = new double[Is],*dI = nullptr,*O,**dO;
    network::Network<double> *network;
    clock_t a,b;
    a = std::clock();
    {
        unsigned l = 0;
        layer::Layer<double> **layers = new layer::Layer<double>*[N_CONV*2+N_FCL];
        layers[l] = new conv::OMPLayer<double>(Ir,Ic,N_CH*scale,
                                               Fr,Fc,1,
                                               Pr,Pc,1,1,LR,
                                               &I,&dI);
        ((conv::OMPLayer<double>*)layers[l])->init();
        while(l < 2*(N_CONV-1))
        {
            ++l;
            layers[l] = new pool::OMPLayer<double>(Ir,Ic,N_CH,
                                                   1,1,scale,
                                                   &layers[l-1]->O,
                                                   &layers[l-1]->dO);
            ++l;
            layers[l] = new conv::OMPLayer<double>(Ir,Ic,N_CH*scale,
                                                   Fr,Fc,N_CH,
                                                   Pr,Pc,1,1,LR,
                                                   &layers[l-1]->O,
                                                   &layers[l-1]->dO);
            ((conv::OMPLayer<double>*)layers[l])->init();
        }
        ++l;
        layers[l] = new pool::OMPLayer<double>(Ir,Ic,1,
                                               1,1,N_CH*scale,
                                               &layers[l-1]->O,
                                               &layers[l-1]->dO);
        ++l;
        layers[l] = new fcl::OMPLayer<double>(Is,FCL_SIZE,
                                              &layers[l-1]->O,
                                              &layers[l-1]->dO,
                                              LR);
        ((fcl::OMPLayer<double>*)layers[l])->init();
        for(unsigned f = 1;f < N_FCL-1;++f)
        {
            ++l;
            layers[l] = new fcl::OMPLayer<double>(FCL_SIZE,FCL_SIZE,
                                                  &layers[l-1]->O,
                                                  &layers[l-1]->dO,
                                                  LR);
            ((fcl::OMPLayer<double>*)layers[l])->init();
        }
        ++l;
        layers[l] = new fcl::OMPLayer<double>(FCL_SIZE,10,
                                              &layers[l-1]->O,
                                              &layers[l-1]->dO,
                                              LR);
        ((fcl::OMPLayer<double>*)layers[l])->init();
        dO = &layers[l]->dO;
        O = layers[l]->O;
        network = new network::Network<double>(layers,l+1);
    }
    b = std::clock();
    out.setup = (double)(b - a) / CLOCKS_PER_SEC;
    unsigned *order_train,*order_test;
    orders(&order_train,&order_test);
    bool first = true;
    clock_t maxF,minF,maxB,minB,diff;
    out.avgF = out.avgB = 0;
    for(unsigned epoch = 0;epoch < N_EPOCHS;++epoch)
    {
        shuffle(order_train,nTrain);
        for(unsigned i = 0;i < TRAIN_PER_EPOCH;++i)
        {
            image(train_images,I,order_train[i]);

            a = std::clock();
            network->forward();
            b = std::clock();
            diff = b - a;
            if(first) maxF = minF = diff;
            else
            {
                maxF = std::max(maxF,diff);
                minF = std::min(minF,diff);
            }
            out.avgF += (double)diff/TRAIN_PER_EPOCH/N_EPOCHS/CLOCKS_PER_SEC;

            eval(label(train_labels,order_train[i]),O,*dO = new double[10]);

            a = std::clock();
            network->backward();
            b = std::clock();
            diff = b - a;
            if(first) maxB = minB = diff;
            else
            {
                maxB = std::max(maxB,diff);
                minB = std::min(minB,diff);
            }
            out.avgB += (double)diff/TRAIN_PER_EPOCH/N_EPOCHS/CLOCKS_PER_SEC;

            delete[] dI;
            first = false;
        }
        shuffle(order_test,nTest);
        unsigned correct = 0;
        for(unsigned i = 0;i < TEST_PER_EPOCH;++i)
        {
            image(test_images,I,order_test[i]);
            network->forward();
            correct += check(O,label(test_labels,order_test[i]));
        }
        test_results(epoch,correct);
    }
    network_result(*network,I,O);
    out.highF = (double)maxF / CLOCKS_PER_SEC;
    out.highB = (double)maxB / CLOCKS_PER_SEC;
    out.lowF = (double)minF / CLOCKS_PER_SEC;
    out.lowB = (double)minB / CLOCKS_PER_SEC;
    a = std::clock();
    delete network;
    delete[] order_train,order_test,I;
    b = std::clock();
    out.tearDown = (double)(b - a) / CLOCKS_PER_SEC;
    return out;
}
mnist::Record mnist::runGPU(unsigned scale)
{
    Record out{0,0,0,0,0,0,0,0};
    if(!nTrain) return out;
    scale += 2;
    constexpr const unsigned Fr = FILTER_SIZE,Fc = FILTER_SIZE,
                             Pr = (Fr-1)/2,Pc = (Fc-1)/2;
    double *I,*d_I,*dI = nullptr,*O,**dO,*d_O,*h_dO;
    clock_t a,b;
    a = std::clock();
    GPU::allocHostPinned(&I,Is);
    GPU::allocHostPinned(&h_dO,10);
    cudaStream_t stream = GPU::createStream();
    network::Network<double> *network;
    {
        unsigned l = 0;
        layer::Layer<double> **layers = new layer::Layer<double>*[N_CONV*2+N_FCL];
        layers[l] = new conv::GPULayer<double>(Ir,Ic,N_CH*scale,
                                               Fr,Fc,1,
                                               Pr,Pc,1,1,LR,
                                               &I,&dI,
                                               stream,
                                               &d_I);
        ((conv::GPULayer<double>*)layers[l])->init();
        while(l < 2*(N_CONV-1))
        {
            ++l;
            layers[l] = new pool::GPULayer<double>(Ir,Ic,N_CH,
                                                   1,1,scale,
                                                   &layers[l-1]->O,
                                                   &layers[l-1]->dO,
                                                   stream,
                                                   &((conv::GPULayer<double>*)layers[l-1])->d_O);
            ++l;
            layers[l] = new conv::GPULayer<double>(Ir,Ic,N_CH*scale,
                                                   Fr,Fc,N_CH,
                                                   Pr,Pc,1,1,LR,
                                                   &layers[l-1]->O,
                                                   &layers[l-1]->dO,
                                                   stream,
                                                   &((pool::GPULayer<double>*)layers[l-1])->d_O);
            ((conv::GPULayer<double>*)layers[l])->init();
        }
        ++l;
        layers[l] = new pool::GPULayer<double>(Ir,Ic,1,
                                               1,1,N_CH*scale,
                                               &layers[l-1]->O,
                                               &layers[l-1]->dO,
                                               stream,
                                               &((conv::GPULayer<double>*)layers[l-1])->d_O);
        ++l;
        layers[l] = new fcl::GPULayer<double>(Is,FCL_SIZE,
                                              &layers[l-1]->O,
                                              &layers[l-1]->dO,
                                              LR,
                                              &((pool::GPULayer<double>*)layers[l-1])->d_O,
                                              stream);
        ((fcl::GPULayer<double>*)layers[l])->init();
        for(unsigned f = 1;f < N_FCL-1;++f)
        {
            ++l;
            layers[l] = new fcl::GPULayer<double>(FCL_SIZE,FCL_SIZE,
                                                  &layers[l-1]->O,
                                                  &layers[l-1]->dO,
                                                  LR,
                                                  &((fcl::GPULayer<double>*)layers[l-1])->d_O,
                                                  stream);
            ((fcl::GPULayer<double>*)layers[l])->init();
        }
        ++l;
        layers[l] = new fcl::GPULayer<double>(FCL_SIZE,10,
                                              &layers[l-1]->O,
                                              &layers[l-1]->dO,
                                              LR,
                                              &((fcl::GPULayer<double>*)layers[l-1])->d_O,
                                              stream);
        ((fcl::GPULayer<double>*)layers[l])->init();
        dO = &layers[l]->dO;
        O = layers[l]->O;
        d_O = ((fcl::GPULayer<double>*)layers[l])->d_O;
        network = new network::Network<double>(layers,l+1);
    }
    b = std::clock();
    out.setup = (double)(b - a)/CLOCKS_PER_SEC;
    unsigned *order_train,*order_test;
    orders(&order_train,&order_test);
    bool first = true;
    clock_t maxF,minF,maxB,minB,diff;
    out.avgF = out.avgB = 0;
    for(unsigned epoch = 0;epoch < N_EPOCHS;++epoch)
    {
        shuffle(order_train,nTrain);
        for(unsigned i = 0;i < TRAIN_PER_EPOCH;++i)
        {
            image(train_images,I,order_train[i]);
            GPU::allocTransfer(I,&d_I,Is,stream);

            a = std::clock();
            network->forward();
            b = std::clock();
            diff = b - a;
            if(first) maxF = minF = diff;
            else
            {
                maxF = std::max(maxF,diff);
                minF = std::min(minF,diff);
            }
            out.avgF += (double)diff/TRAIN_PER_EPOCH/N_EPOCHS/CLOCKS_PER_SEC;

            GPU::destroyDeviceMem(d_O,stream);
            GPU::sync(stream);

            eval(label(train_labels,order_train[i]),O,h_dO);
            GPU::allocTransfer(h_dO,dO,10,stream);

            a = std::clock();
            network->backward();
            b = std::clock();
            diff = b - a;
            if(first) maxB = minB = diff;
            else
            {
                maxB = std::max(maxB,diff);
                minB = std::min(minB,diff);
            }
            out.avgB += (double)diff/TRAIN_PER_EPOCH/N_EPOCHS/CLOCKS_PER_SEC;

            GPU::destroyDeviceMem(dI,stream);
            first = false;
        }
        shuffle(order_test,nTest);
        unsigned correct = 0;
        for(unsigned i = 0;i < TEST_PER_EPOCH;++i)
        {
            image(test_images,I,order_test[i]);
            GPU::allocTransfer(I,&d_I,Is,stream);

            network->forward();

            GPU::destroyDeviceMem(d_O,stream);
            GPU::sync(stream);
            correct += check(O,label(test_labels,order_test[i]));
        }
        test_results(epoch,correct);
    }
    gpu_result(*network,I,&d_I,stream,d_O,O);
    out.highF = (double)maxF / CLOCKS_PER_SEC;
    out.highB = (double)maxB / CLOCKS_PER_SEC;
    out.lowF = (double)minF / CLOCKS_PER_SEC;
    out.lowB = (double)minB / CLOCKS_PER_SEC;
    a = std::clock();
    GPU::destroyStream(stream);
    GPU::destroyHostPinned(h_dO);
    GPU::destroyHostPinned(I);
    delete network;
    delete[] order_train,order_test;
    b = std::clock();
    out.tearDown = (double)(b - a) / CLOCKS_PER_SEC;
    GPU::sync();
    GPU::reset();
    return out;
}

void dbg_erf(unsigned label)
{
    double O[10]{},dO[10]{};
    O[label] = 1;
    eval(label,O,dO);
    printf(" O:[");
    for(unsigned o = 0;o < 10;++o)
    {
        printf(" %f",O[o]);
        O[o] = !(o==label);
    }
    printf(" ]\ndO:[");
    for(unsigned o = 0;o < 10;++o)
    {
        printf(" %f",dO[o]);
        dO[o] = 0;
    }
    printf("]\n\n");
    eval(label,O,dO);
    printf(" O:[");
    for(unsigned o = 0;o < 10;++o)
        printf(" %f",O[o]);
    printf(" ]\ndO:[");
    for(unsigned o = 0;o < 10;++o)
        printf(" %f",dO[o]);
    printf("]\n\n");

}
void mnist::dbg()
{
    srand((int)time(NULL));
    {
        unsigned img = rand() % nTrain;
        double *I = new double[Is];
        image(train_images,I,img);
        dbg_img(I);
        unsigned lbl = label(train_labels,img);
        printf("train label: %u\n\n",lbl);
        delete[] I;
        dbg_erf(lbl);
    }
    {
        unsigned img = rand() % nTest;
        double *I = new double[Is];
        image(test_images,I,img);
        dbg_img(I);
        unsigned lbl = label(test_labels,img);
        printf("test label: %u\n\n",lbl);
        delete[] I;
        dbg_erf(lbl);
    }
}

void mnist::memcheck_gpu()
{
    constexpr const unsigned Fr = FILTER_SIZE,Fc = FILTER_SIZE,
                             Pr = (Fr-1)/2,Pc = (Fc-1)/2;
    {
        double *I,*d_I,*dI = nullptr,*O,**dO,*d_O,*h_dO;
        cudaStream_t stream = GPU::createStream();
        network::Network<double> *network;
        {
            unsigned l = 0;
            layer::Layer<double> **layers = new layer::Layer<double>*[N_CONV * 2 + N_FCL];
            layers[l] = new conv::GPULayer<double>(Ir,Ic,N_CH * 2,
                                                   Fr,Fc,1,
                                                   Pr,Pc,1,1,LR,
                                                   &I,&dI,
                                                   stream,
                                                   &d_I);
            ((conv::GPULayer<double>*)layers[l])->init();
            while(l < 2 * (N_CONV - 1))
            {
                ++l;
                layers[l] = new pool::GPULayer<double>(Ir,Ic,N_CH,
                                                       1,1,2,
                                                       &layers[l - 1]->O,
                                                       &layers[l - 1]->dO,
                                                       stream,
                                                       &((conv::GPULayer<double>*)layers[l - 1])->d_O);
                ++l;
                layers[l] = new conv::GPULayer<double>(Ir,Ic,N_CH * 2,
                                                       Fr,Fc,N_CH,
                                                       Pr,Pc,1,1,LR,
                                                       &layers[l - 1]->O,
                                                       &layers[l - 1]->dO,
                                                       stream,
                                                       &((pool::GPULayer<double>*)layers[l - 1])->d_O);
                ((conv::GPULayer<double>*)layers[l])->init();
            }
            ++l;
            layers[l] = new pool::GPULayer<double>(Ir,Ic,1,
                                                   1,1,N_CH * 2,
                                                   &layers[l - 1]->O,
                                                   &layers[l - 1]->dO,
                                                   stream,
                                                   &((conv::GPULayer<double>*)layers[l - 1])->d_O);
            ++l;
            layers[l] = new fcl::GPULayer<double>(Is,FCL_SIZE,
                                                  &layers[l - 1]->O,
                                                  &layers[l - 1]->dO,
                                                  LR,
                                                  &((pool::GPULayer<double>*)layers[l - 1])->d_O,
                                                  stream);
            ((fcl::GPULayer<double>*)layers[l])->init();
            for(unsigned f = 1;f < N_FCL - 1;++f)
            {
                ++l;
                layers[l] = new fcl::GPULayer<double>(FCL_SIZE,FCL_SIZE,
                                                      &layers[l - 1]->O,
                                                      &layers[l - 1]->dO,
                                                      LR,
                                                      &((fcl::GPULayer<double>*)layers[l - 1])->d_O,
                                                      stream);
                ((fcl::GPULayer<double>*)layers[l])->init();
            }
            ++l;
            layers[l] = new fcl::GPULayer<double>(FCL_SIZE,10,
                                                  &layers[l - 1]->O,
                                                  &layers[l - 1]->dO,
                                                  LR,
                                                  &((fcl::GPULayer<double>*)layers[l - 1])->d_O,
                                                  stream);
            ((fcl::GPULayer<double>*)layers[l])->init();
            dO = &layers[l]->dO;
            O = layers[l]->O;
            d_O = ((fcl::GPULayer<double>*)layers[l])->d_O;
            network = new network::Network<double>(layers,l + 1);
        }
        GPU::allocHostPinned(&I,FCL_SIZE);
        GPU::allocHostPinned(&h_dO,10);

        image(train_images,I,0);
        GPU::allocTransfer(I,&d_I,Is,stream);

        network->forward();

        GPU::destroyDeviceMem(d_O,stream);
        GPU::sync(stream);

        eval(label(train_labels,0),O,h_dO);
        GPU::allocTransfer(h_dO,dO,10,stream);

        network->backward();

        GPU::destroyDeviceMem(dI,stream);
        gpu_result(*network,I,&d_I,stream,d_O,O);
        GPU::destroyStream(stream);
        GPU::destroyHostPinned(h_dO);
        GPU::destroyHostPinned(I);
        delete network;
    }
    GPU::sync();
    GPU::reset();
}