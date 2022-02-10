#ifndef LAYER
#define LAYER
#include <cl_math_lib.cpp>
#include <vector>
#include <enums.cpp>
#include <math.h>
template <class T>
class Layer {
    public:
        Layer* next;
        Layer* prev;

        vector<T> neurons;
        vector<T> intermed;
        vector<T> errors;
        vector<T> weights;
        vector<T> bias   ;
        vector<T> dW     ;
        vector<T> dWBias ;

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

        

        virtual void   closs(vector<float> &target) = 0;
        virtual void   fwd()   = 0;
        virtual void   bwd()   = 0;
        virtual void   learn(float learnRate) = 0;
        virtual Layer* clone() = 0;
};

#endif