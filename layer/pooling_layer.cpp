#ifndef CONV2D
#define CONV2D

#include "basic_layer.cpp"

class Pooling: public Layer {
    public:
        Pooling(){ this -> type = PoolingLayer; };
        Pooling(array<size_t, 3> inpDims, std::array<size_t, 2> poolingDims, size_t convolutions, Activation iAct, PoolingType poolingtype);
        Pooling(std::array<size_t, 2> poolingDims, size_t convolutions, Activation iAct, PoolingType poolingtype);
        Pooling(Pooling & other);
        Pooling(Pooling &&other);
        ~Pooling();

        Pooling& operator=(Pooling other);
        shared_ptr<Layer> clone();

        void fwd();
        void bwd();
        void learn(const float learnRate);
        void closs(vector<float> &target);
        void setupLayer();
        void setInput(vector<float> & inp);
        void summary();

    private:
        void copyContent(Pooling& other);
        void swap(Pooling& other);
        void evalOptimizer();

        vector<float> maxPoolIndizies;
        std::array<size_t, 2> poolingDims;
        array<size_t, 3> outDims;
        PoolingType poolingtype;


    private:
        std::shared_ptr<ClMathLib> mathLib;
};

Pooling::Pooling(array<size_t, 3>inpDims, std::array<size_t, 2> poolingDims, size_t convolutions, Activation iAct, PoolingType poolingtype = MAX)
    :Layer()
    {
    this -> activation   = iAct;
    this -> mathLib      = ClMathLib::instanceML();
    this -> poolingDims   = poolingDims;
    this -> inpDims      = inpDims;
    this -> type         = PoolingLayer;
    this -> poolingtype  = poolingtype;
    this -> evalOptimizer();
}

Pooling::Pooling(std::array<size_t, 2> poolingDims, size_t convolutions, Activation iAct, PoolingType poolingtype = MAX)
    :Layer()
    {
    this -> activation   = iAct;
    this -> mathLib      = ClMathLib::instanceML();
    this -> poolingDims   = poolingDims;
    this -> type         = PoolingLayer;
    this -> inpDims      = array<size_t, 3> ();
    this -> poolingtype  = poolingtype;
    this -> evalOptimizer();
}

//assignment, copy and move constructor and belonging methods

Pooling& Pooling::operator=(Pooling other) {
   this -> swap();
   return *this;
}


Pooling::~Pooling() {
   /* if(prev != nullptr)
        delete prev;
    if(next != nullptr)
        delete next;*/
}

Pooling::Pooling(Pooling &other): Layer() {
    //std::allocator<Layer> a;
    //prev = a.allocate(sizeof(*prev));
    if(other.prev != nullptr) {
        this -> prev = other.prev -> clone();
    }
    if(other.next != nullptr) {
        this -> next = other.next -> clone();
    }
    copyContent(other);
 }


Pooling::Pooling(Pooling&& other): Layer() {
    this -> swap(other);
 }

shared_ptr<Layer> Pooling::clone() {
    return shared_ptr<Pooling>(new Pooling(*this));
}

void Pooling::copyContent(Pooling& other) {
    this -> errors         = other.errors;
    this -> neurons        = other.neurons;
    //lossfunction   = other.lossfunction;
    this -> loss           = other.loss;
    this -> activation     = other.activation;
    this -> weights        = other.weights;
    this -> weight_width   = other.weight_width;
    this -> inpDims        = other.inpDims;
    this -> outDims        = other.outDims;
    this -> bias           = other.bias;
    this -> intermed       = other.intermed;
    this -> mathLib        = other.mathLib;
    this -> type           = ConvolutionalLayer;
    this -> optimizer      = other.optimizer;
 }

void Pooling::swap(Pooling& other) {
    using std::swap;
    swap(this -> errors       , other.errors);
    swap(this -> neurons      , other.neurons);
    swap(this -> prev         , other.prev);
    swap(this -> bias         , other.bias);
    swap(this -> intermed     , other.intermed);
    swap(this -> inpDims      , other.inpDims);
    swap(this -> outDims      , other.outDims);
    swap(this -> next         , other.next);
    swap(this -> prev         , other.prev);
    //lossfunction   = other.lossfunction;
    this -> activation     = other.activation;
    this -> mathLib        = other.mathLib;
    this -> type           = OutputLayer;
    this -> optimizer      = other.optimizer;
    this -> weight_width   = other.weight_width;
    this -> type           = DenseLayer;
 }

void Pooling::setupLayer() {
    if( this->next == nullptr ) {
         cout << "Network not initializied properly \n";
         throw new std::logic_error("Structure missmatch");
    }

    size_t layerFrom = this -> inpDims[0] * this -> inpDims[1] * this -> inpDims[2];
    const uint8_t bordX = this -> inpDims[0] % this -> poolingDims[0] == 0 ? 0 : 1;
    const uint8_t bordY = this -> inpDims[1] % this -> poolingDims[1] == 0 ? 0 : 1;

    const size_t newDimsX = ( this -> inpDims[0] / this -> poolingDims[0] ) + bordX;
    const size_t newDimsY = ( this -> inpDims[1] / this -> poolingDims[1] ) + bordY;

    const size_t channels     = this -> inpDims[2]; 
    
    this -> outDims = { newDimsX, newDimsY, channels};

    if( this -> next -> type == ConvolutionalLayer ||  this -> next -> type == PoolingLayer ) {
        this -> next -> inpDims = this -> outDims;
    }
    else {
        this -> next -> neurons = vector<float>( newDimsX * newDimsY * channels );
    }
    const size_t layerTo      = newDimsX * newDimsY * channels;

    this -> neurons         = vector<float>( layerFrom );

    this -> errors          = vector<float>( layerFrom );
    this -> weights         = vector<float>(0);

    this -> next -> setupLayer();
}

void Pooling::setInput(vector<float> & inp) {
    this -> neurons = inp;
};

void Pooling::fwd() {
    
    this->next->fwd();
}

void Pooling::bwd() {
    //calc dWs
    array<size_t, 3> outD = {  this -> poolingDims[0], this ->poolingDims[1], this -> outDims[2]};
    
    this -> prev -> bwd();
}

void Pooling::learn(const float learnRate) {
    this->next->learn(learnRate);
}


void Pooling::closs(vector<float> &target) {
    return;
}

void Pooling::summary() {
            const string type = this -> type == ConvolutionalLayer ? "Convolutional Layer" : this -> type == DenseLayer ?  "Dense Layer" : this -> type == PoolingLayer ? "Pooling Layer" : "Dense Output Layer";
            cout << type << ": \n";
            cout << "Neurons:\t\t" << this -> neurons.size()  << "\n" ;
            cout << "Poolings: \t\t" << this -> inpDims[2] <<" per " << this -> poolingDims[0] << " x " << this -> poolingDims[1] <<  "\n";
            cout << "Total trainable params: 0" << "\n"; 
            cout << "Output dims:\t"      << this -> outDims[0] << " x " << this -> outDims[1] <<  "\n";
            cout << "Input channels:\t\t" << this -> inpDims[2] <<" per " << this -> inpDims[0] << " x " << this -> inpDims[1] <<  "\n";
            cout << "_______________________________________________________________________________________\n\n";
            if ( this -> next != nullptr ) {
                this -> next -> summary();
            }
        }
#endif