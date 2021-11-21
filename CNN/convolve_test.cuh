#pragma once

#define DEBUG

#include "convolve.cuh"

namespace convolve_test
{
    void testSTC();
    void testOMP();
    void testGPU();

    void dbg();
}