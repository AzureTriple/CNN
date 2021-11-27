#pragma once

namespace mnist
{
    struct Record
    {
        double setup,tearDown,
               highF,lowF,avgF,
               highB,lowB,avgB;
    };

    Record runSTC(unsigned scale);
    Record runOMP(unsigned scale);
    Record runGPU(unsigned scale);

    int init(const char *train_img,const char *train_lbl,
             const char *test_img,const char *test_lbl,
             unsigned epochs,unsigned train_count,
             unsigned test_count,double lr);
    void close();

    void dbg();
    void memcheck_gpu();
}