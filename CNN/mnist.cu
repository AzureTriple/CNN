#include "mnist.cuh"
#include "convolve.cuh"
#include "pool.cuh"
#include "fcl.cuh"
#include "network.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>

std::ifstream train_labels("C:/Users/prgmPC/Desktop/School/Y4/ee/CNN/train-labels.idx1-ubyte",std::ifstream::binary);
std::ifstream train_images("C:/Users/prgmPC/Desktop/School/Y4/ee/CNN/train-images.idx3-ubyte",std::ifstream::binary);
std::ifstream test_labels("C:/Users/prgmPC/Desktop/School/Y4/ee/CNN/t10k-labels.idx1-ubyte",std::ifstream::binary);
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
    return *(T *)(void *)buf;
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
    for(unsigned o = 0;o < 10;++o)
        s += O[o] = std::exp(std::max<double>(O[o] - max,EXP_MIN));
    for(unsigned o = 0;o < 10;++o)
    {
        O[o] /= s;
        dO[o] = O[o]-!(o!=expected);
        //dO[o] = O[o]*O[o]*(1-O[o]);
    }
    //dO[expected] += O[expected]*(O[expected]-1);
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
    printf("max: %f | min: %f | avg: %f\n",maxv,minv,avgv/Ir/Ic);
}

constexpr const unsigned N_EPOCHS = 1<<6,N_CONV = 4,N_FCL = 4,
                         FILTER_SIZE = 15,FCL_SIZE = 1024,N_CH = 4,
                         TRAIN_PER_EPOCH = 1000,TEST_PER_EPOCH = 100;
constexpr const double LR = 0.01;

int mnist::runSTC(unsigned scale)
{
    if(!nTrain) return 1;
    scale += 2;
    constexpr const unsigned Fr = FILTER_SIZE,Fc = FILTER_SIZE,
                             Pr = (Fr-1)/2,Pc = (Fc-1)/2;
    double *I = new double[Ir*Ic],*dI,**O,**dO;
    network::Network<double> *network;
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
        layers[l] = new fcl::STCLayer<double>(Ir*Ic,FCL_SIZE,
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
        O = &layers[l]->O;
        network = new network::Network<double>(layers,l+1);
    }
    unsigned *order = new unsigned[nTrain];
    for(unsigned i = 0;i < nTrain;++i)
        order[i] = i;
    unsigned *order_test = new unsigned[nTest];
    for(unsigned i = 0;i < nTest;++i)
        order_test[i] = i;
    for(unsigned epoch = 0;epoch < N_EPOCHS;++epoch)
    {
        shuffle(order,nTrain);
        for(unsigned i = 0;i < TRAIN_PER_EPOCH;++i)
        {
            image(train_images,I,order[i]);
            network->forward();
            eval(label(train_labels,order[i]),*O,*dO = new double[10]);
            network->backward();
            delete[] dI;
        }
        shuffle(order_test,nTest);
        unsigned correct = 0;
        for(unsigned i = 0;i < TEST_PER_EPOCH;++i)
        {
            image(test_images,I,order_test[i]);
            network->forward();
            softmax(*O);
            double max = (*O)[0];
            unsigned midx = 0;
            for(unsigned j = 1;j < 10;++j)
            {
                if(max < (*O)[j])
                {
                    max = (*O)[j];
                    midx = j;
                }
            }
            correct += !(midx != label(test_labels,order_test[i]));
        }
        printf(
            "epoch %2u: %3u/%3u = %5.2f%%\n",
            epoch,
            correct,
            TEST_PER_EPOCH,
            100.*correct/TEST_PER_EPOCH
        );
    }
    delete network;
    delete[] order,I;
    return 0;
}
int mnist::runOMP(unsigned scale)
{
    if(!nTrain) return 1;
    scale += 2;
    constexpr const unsigned Fr = FILTER_SIZE,Fc = FILTER_SIZE,
                             Pr = (Fr-1)/2,Pc = (Fc-1)/2;
    double *I = new double[Ir*Ic],*dI,**O,**dO;
    network::Network<double> *network;
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
        layers[l] = new fcl::OMPLayer<double>(Ir*Ic,FCL_SIZE,
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
        O = &layers[l]->O;
        network = new network::Network<double>(layers,l+1);
    }
    unsigned *order_train = new unsigned[nTrain];
    for(unsigned i = 0;i < nTrain;++i)
        order_train[i] = i;
    unsigned *order_test = new unsigned[nTest];
    for(unsigned i = 0;i < nTest;++i)
        order_test[i] = i;
    for(unsigned epoch = 0;epoch < N_EPOCHS;++epoch)
    {
        shuffle(order_train,nTrain);
        for(unsigned i = 0;i < TRAIN_PER_EPOCH;++i)
        {
            image(train_images,I,order_train[i]);
            network->forward();
            eval(label(train_labels,order_train[i]),*O,*dO = new double[10]);
            network->backward();
            delete[] dI;
        }
        shuffle(order_test,nTest);
        unsigned correct = 0;
        for(unsigned i = 0;i < TEST_PER_EPOCH;++i)
        {
            image(test_images,I,order_test[i]);
            network->forward();
            softmax(*O);
            double max = (*O)[0];
            unsigned midx = 0;
            for(unsigned j = 1;j < 10;++j)
            {
                if(max < (*O)[j])
                {
                    max = (*O)[j];
                    midx = j;
                }
            }
            if(midx == label(test_labels,order_test[i]))
                ++correct;
        }
        printf(
            "epoch %2u: %3u/%3u = %5.2f%%\n",
            epoch,
            correct,
            TEST_PER_EPOCH,
            100.*correct/TEST_PER_EPOCH
        );
    }
    {
        printf("final:\n");
        unsigned img = rand() % nTest;
        image(test_images,I,img);
        dbg_img(I);

        network->forward();

        softmax(*O);
        printf("out:[%f",(*O)[0]);
        double max = (*O)[0];
        unsigned midx = 0;
        for(unsigned j = 1;j < 10;++j)
        {
            printf(" %f",(*O)[j]);
            if(max < (*O)[j])
            {
                max = (*O)[j];
                midx = j;
            }
        }
        printf("]\nlabel: %u\npredicted: %u\n",label(test_labels,img),midx);
    }
    delete network;
    delete[] order_train,I;
    return 0;
}
int mnist::runGPU(unsigned scale)
{
    if(!nTrain) return 1;
    scale += 2;
    constexpr const unsigned Fr = FILTER_SIZE,Fc = FILTER_SIZE,
                             Pr = (Fr-1)/2,Pc = (Fc-1)/2;
    double *I,*d_I,*dI,**O,**dO,**d_O,*h_dO;
    GPU::allocHostPinned(&I,Ir*Ic);
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
        layers[l] = new fcl::GPULayer<double>(Ir*Ic,FCL_SIZE,
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
        O = &layers[l]->O;
        d_O = &((fcl::GPULayer<double>*)layers[l])->d_O;
        network = new network::Network<double>(layers,l+1);
    }
    unsigned *order_train = new unsigned[nTrain];
    for(unsigned i = 0;i < nTrain;++i)
        order_train[i] = i;
    unsigned *order_test = new unsigned[nTest];
    for(unsigned i = 0;i < nTest;++i)
        order_test[i] = i;
    for(unsigned epoch = 0;epoch < N_EPOCHS;++epoch)
    {
        shuffle(order_train,nTrain);
        for(unsigned i = 0;i < TRAIN_PER_EPOCH;++i)
        {
            image(train_images,I,order_train[i]);
            GPU::allocTransfer(I,&d_I,Ir*Ic,stream);

            network->forward();

            GPU::destroyDeviceMem(*d_O,stream);
            GPU::sync(stream);

            eval(label(train_labels,order_train[i]),*O,h_dO);

            GPU::allocTransfer(h_dO,dO,10,stream);

            network->backward();

            GPU::destroyDeviceMem(dI,stream);
        }
        shuffle(order_test,nTest);
        unsigned correct = 0;
        for(unsigned i = 0;i < TEST_PER_EPOCH;++i)
        {
            image(test_images,I,order_test[i]);
            GPU::allocTransfer(I,&d_I,Ir*Ic,stream);

            network->forward();

            GPU::destroyDeviceMem(*d_O,stream);
            GPU::sync(stream);
            softmax(*O);
            double max = (*O)[0];
            unsigned midx = 0;
            for(unsigned j = 1;j < 10;++j)
            {
                if(max < (*O)[j])
                {
                    max = (*O)[j];
                    midx = j;
                }
            }
            if(midx == label(test_labels,order_test[i]))
                ++correct;
        }
        printf(
            "epoch %2u: %3u/%3u = %5.2f%%\n",
            epoch,
            correct,
            TEST_PER_EPOCH,
            100.*correct/TEST_PER_EPOCH
        );
    }
    {
        printf("final:\n");
        unsigned img = rand() % nTest;
        image(test_images,I,img);
        dbg_img(I);
        GPU::allocTransfer(I,&d_I,Ir*Ic,stream);

        network->forward();

        GPU::destroyDeviceMem(*d_O,stream);
        GPU::sync(stream);
        softmax(*O);

        printf("out:[%f",(*O)[0]);
        double max = (*O)[0];
        unsigned midx = 0;
        for(unsigned j = 1;j < 10;++j)
        {
            printf(" %f",(*O)[j]);
            if(max < (*O)[j])
            {
                max = (*O)[j];
                midx = j;
            }
        }
        printf("]\nlabel: %u\npredicted: %u\n",label(test_labels,img),midx);
    }

    GPU::destroyHostPinned(h_dO);
    GPU::destroyHostPinned(I);
    GPU::sync(stream);
    GPU::destroyStream(stream);
    GPU::sync();
    delete network;
    delete[] order_train;
    return 0;
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
        double *I = new double[Ir*Ic];
        image(train_images,I,img);
        dbg_img(I);
        unsigned lbl = label(train_labels,img);
        printf("train label: %u\n\n",lbl);
        delete[] I;
        dbg_erf(lbl);
    }
    {
        unsigned img = rand() % nTest;
        double *I = new double[Ir*Ic];
        image(test_images,I,img);
        dbg_img(I);
        unsigned lbl = label(test_labels,img);
        printf("test label: %u\n\n",lbl);
        delete[] I;
        dbg_erf(lbl);
    }
}