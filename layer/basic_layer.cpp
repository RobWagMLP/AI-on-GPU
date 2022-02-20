#ifndef LAYER
#define LAYER

#include "enums.cpp"
#include "./../cl_math_lib/cl_math_lib.cpp"
#include <vector>
#include <math.h>

class Layer {
    public:

        LayerType type;
        Layer* next;
        Layer* prev;

        vector<float> neurons;
        vector<float> intermed;
        vector<float> errors;
        vector<float> weights;
        vector<float> bias   ;

        Layer(uint32_t inp_size):errors(inp_size), neurons(inp_size) {
            next = nullptr;
            prev = nullptr;
        };

        Layer() {
            next = nullptr;
            prev = nullptr;
        };

        Loss loss             ;
        Activation activation ;
        size_t      weight_width;

        std::function<void(vector<float>&)> lossfunction;
        vector<size_t> inpDims;       

        virtual void   closs(vector<float> &target) = 0;
        virtual void   fwd()   = 0;
        virtual void   bwd()   = 0;
        virtual void   learn(const float learnRate) = 0;
        virtual void   setupLayer() = 0;
        virtual Layer* clone() = 0;
};

#endif