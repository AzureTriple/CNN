#pragma once

#include "layer.h"

namespace network
{
    using layer::Layer;

    template<typename T>
    struct Network
    {
        const unsigned nL;
        Layer<T> **layers;

        Network(Layer<T> **layers,const unsigned nL)
            : nL(nL),layers(layers) {}

        virtual ~Network()
        {
            for(unsigned l = 0;l < nL;++l)
                delete layers[l];
            delete[] layers;
        }

        void forward()
        {
            for(unsigned l = 0;l < nL;++l)
                layers[l]->forward();
        }
        void backward()
        {
            for(unsigned l = nL;l;)
                layers[--l]->backward();
        }
    };
}