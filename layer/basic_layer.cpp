#include <cl_math_lib.cpp>
#include <vector>
#include <enums.cpp>
#include <math.h>

class Layer {
    public:
        Layer* next;
        Layer* prev;

        vector<float> neurons;
        vector<float> intermed;
        vector<float> errors;
        vector<float> weights;
        vector<float> bias   ;
        vector<float> dW     ;

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
        virtual void   learn() = 0;
        virtual Layer* clone() = 0;
};