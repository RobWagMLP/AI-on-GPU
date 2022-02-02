#include <cl_math_lib.cpp>
#include <vector>
#include <enums.cpp>

class Layer {
    public:
        Layer* next;
        Layer* prev;
        vector<float>* neurons;
        vector<float>* errors;
        vector<float>* weights;
        vector<float>* bias   ;
        vector<float>* dW     ;
        Loss loss             ;
        Activation activation ;
        size_t      weight_width;

    virtual void closs() = 0;
    virtual void fwd()   = 0;
    virtual void bwd()   = 0;
    virtual void learn() = 0;
};