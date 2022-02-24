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
         Layer(uint32_t inp_size):errors(inp_size), neurons(inp_size) {
            next = nullptr;
            prev = nullptr;
        };

        Layer() {
            next = nullptr;
            prev = nullptr;
        };

        vector<float>& getOutput() {
            return this -> neurons;
        };

        virtual void   closs(vector<float> &target) = 0;
        virtual void   fwd()   = 0;
        virtual void   bwd()   = 0;
        virtual void   learn(const float learnRate) = 0;
        virtual void   setupLayer() = 0;
        virtual void   setInput(vector<float> & inp) = 0;

        void   summary() {
            const string type = this -> type == ConvolutionalLayer ? "Convolutional Layer" : this -> type == DenseLayer ? "Dense Layer" : "Dense Output Layer";
            cout << type << ": \n";
            cout << "Neurons : " << this -> neurons.size() << "\n" ;
            cout << "Weights: "  << this -> weights.size() << "\n" ;
            cout << "Biases: "   << this -> bias.size()    << "\n";
            cout << "Total trainable params: " << ( this -> weights.size() + this -> bias.size() ) << "\n"; 
            cout << "_______________________________________________________________________________________\n";
            if ( this -> next != nullptr ) {
                this -> next -> summary();
            }
        }

        Loss loss             ;
        Activation activation ;

    protected:
        friend class Conv2D;
        friend class Dense;
        friend class DenseOut;

        vector<float> neurons;
        vector<float> intermed;
        vector<float> errors;
        vector<float> weights;
        vector<float> bias   ;
        size_t      weight_width;

        std::function<void(vector<float>&)> lossfunction;
        vector<size_t> inpDims;       

        virtual Layer* clone() = 0;
};

#endif