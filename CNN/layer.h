#pragma once

namespace layer
{
    template<typename T>
    struct Layer
    {
        T *O,*dO;
        Layer() : O(nullptr),dO(nullptr) {}

        virtual ~Layer() {}

        virtual void forward() = 0;
        virtual void backward() = 0;
    };
}