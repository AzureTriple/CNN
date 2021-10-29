#pragma once

struct Matrix {
    double *data;
    unsigned ch,r,c;

    Matrix(unsigned ch,unsigned r,unsigned c) : ch(ch),r(r),c(c),data(new double[(size_t)ch*r*c]) {}
    ~Matrix() {delete[] data;}
    double& operator()(unsigned _ch,unsigned _r,unsigned _c) {return data[r*c*_ch+c*_r+_c];}
};