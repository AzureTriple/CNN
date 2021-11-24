#pragma once

namespace mnist
{
    int runSTC(unsigned scale);
    int runOMP(unsigned scale);
    int runGPU(unsigned scale);

    void dbg();
}