//#define DEBUG
//#define MEMCHECK

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#ifdef MATH
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
#elif defined(MEMCHECK)
#include "mnist.cuh"
#include <stdexcept>
#include <iostream>
int main()
{
    try 
    {
        mnist::init("../../train-images.idx3-ubyte",
                    "../../train-labels.idx1-ubyte",
                    "../../t10k-images.idx3-ubyte",
                    "../../t10k-labels.idx1-ubyte",
                    1,1,1,1);
        mnist::memcheck_gpu();
    }
    catch(std::exception e)
    {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
#else
#include <cstddef>
#include "mnist.cuh"
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <fstream>
#include <limits>

unsigned getUnsigned(const char *str,unsigned min)
{
    errno = 0;
    long o(std::strtol(str,nullptr,10));
    char *err = nullptr;
    switch(errno)
    {
        case EDOM  : err = (char*)"Domain Error";     break;
        case ERANGE: err = (char*)"Range Error";      break;
        case EILSEQ: err = (char*)"Invalid Sequence"; break;
        default:
            if((unsigned)-1 < o || o < (long)min)
                err = (char*)"Range Error";
    }
    if(err)
    {
        printf("%s: %s\n",err,str);
        exit(1);
    }
    return (unsigned)o;
}
double getDouble(const char *str)
{
    errno = 0;
    double o(std::strtod(str,nullptr));
    char *err = nullptr;
    switch(errno)
    {
        case EDOM  : err = (char*)"Domain Error";     break;
        case ERANGE: err = (char*)"Range Error";      break;
        case EILSEQ: err = (char*)"Invalid Sequence"; break;
        default: break;
    }
    if(err)
    {
        printf("%s: %s\n",err,str);
        exit(1);
    }
    return o;
}
enum class Mode {stc,omp,gpu};
Mode getMode(const char *str)
{
    Mode o;
    bool err(false);
    switch(str[0])
    {
        case 's': o = Mode::stc; break;
        case 'o': o = Mode::omp; break;
        case 'g': o = Mode::gpu; break;
        default: err = true;
    }
    if(!err)
    {
        char *cmp = nullptr;
        switch(o)
        {
            case Mode::stc: cmp = (char*)"tc"; break;
            case Mode::omp: cmp = (char*)"mp"; break;
            case Mode::gpu: cmp = (char*)"pu"; break;
        }
        err = std::strncmp(str + 1,cmp,3);
    }
    if(err)
    {
        printf("Unknown Mode: %s\n",str);
        exit(1);
    }
    return o;
}

int main(int argc,char *argv[])
{
    /*
    srand((int)time(NULL));
    int ret = 0;
    try {mnist::runGPU(0); std::cout << "\nexecution complete.";}
    catch(std::exception e) {std::cerr << e.what(); ret = 1;}
    std::cin.ignore();
    return ret;
    */
    /*
    mnist::dbg();
    std::cin.ignore();
    */
    unsigned epochs = 64,train_count = 1000,test_count = 100,max_scale = 10,min_scale = 0;
    double LR = 0.01;
    Mode mode;
    const char *output;
    switch(argc)
    {
        case 13:   min_scale = getUnsigned(argv[12],0); [[fallthrough]];
        case 12:   max_scale = getUnsigned(argv[11],0); [[fallthrough]];
        case 11:  test_count = getUnsigned(argv[10],1); [[fallthrough]];
        case 10: train_count = getUnsigned(argv[ 9],1); [[fallthrough]];
        case  9:      epochs = getUnsigned(argv[ 8],1); [[fallthrough]];
        case  8:          LR = getDouble  (argv[ 7]  ); [[fallthrough]];
        case  7:        mode = getMode    (argv[ 2]  );
                      output = argv[1];
                 mnist::init(argv[3],argv[4],argv[5],argv[6],
                             epochs,train_count,test_count,LR);
                 break;
        default:
            printf("Usage: %s <output> <stc|omp|gpu> <train_img> <train_lbl> <test_img> <test_lbl> "
                   "[lr] [epochs] [train_per_epoch] [test_per_epoch] [max_scale] [min_scale]\n",argv[0]);
            return 1;
    }
    std::ofstream data(output);
    data.precision(std::numeric_limits<double>::max_digits10);
    data << "setup,tearDown,highF,lowF,avgF,highB,lowB,avgB" << std::endl;
    for(unsigned scale = min_scale;scale <= max_scale;++scale)
    {
        mnist::Record record;
        try
        {
            switch(mode)
            {
                case Mode::stc: record = mnist::runSTC(scale); break;
                case Mode::omp: record = mnist::runOMP(scale); break;
                case Mode::gpu: record = mnist::runGPU(scale); break;
            }
        }
        catch(std::exception e)
        {
            std::cerr << e.what();
            data.close();
            return 1;
        }
        data << record.setup    << ','
             << record.tearDown << ','
             << record.highF    << ','
             << record.lowF     << ','
             << record.avgF     << ','
             << record.highB    << ','
             << record.lowB     << ','
             << record.avgB     << std::endl;
    }
    data.close();
    return 0;
}
#endif