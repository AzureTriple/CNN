//#define DEBUG

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
#include "mnist.cuh"
#include <cstdlib>
#include <ctime>
#include <iostream>

int main()
{
    srand((int)time(NULL));
    int ret = 0;
    try {mnist::runGPU(0); std::cout << "\nexecution complete.";}
    catch(std::exception e) {std::cerr << e.what(); ret = 1;}
    std::cin.ignore();
    return ret;
    /*
    mnist::dbg();
    std::cin.ignore();
    */
}
#endif