#ifndef DENSEOUT
#define DENSEOUT

#include "basic_layer.cpp"
#include "./../config/constants.h"

class DenseOut: public Layer{
    public:
        DenseOut():Layer(){ };
        DenseOut(uint32_t outputSize,Loss iLoss, Layer *iPrev);
        DenseOut(DenseOut & other);
        DenseOut(DenseOut &&other);
        ~DenseOut();

        DenseOut& operator=(DenseOut other);
        DenseOut* clone();

        void copyContent(DenseOut& other);
        void evalLoss(Activation activation);
        void lossBinCrossSig(vector<float> &target);
        void lossBinCrossTan(vector<float> &target);
        void lossBinCrossRel(vector<float> &target);

        void lossSparseCrossSmax(vector<float> &target);
        void lossSparseCrossSig(vector<float> &target);
        void lossSparseCrossTan(vector<float> &target);
        void lossSparseCrossRel(vector<float> &target);
       
        void fwd();
        void bwd();
        void learn(const  float learnRate);
        void closs( vector<float> &target);
        void setupLayer();

        std::function<void(vector<float>&)> lossfunction;
    private:
        std::shared_ptr<ClMathLib> mathLib;
};

DenseOut::DenseOut(uint32_t outputSize, Loss iLoss, Layer *iPrev)
    :Layer(outputSize){
    loss        = iLoss;
    prev        = iPrev; 
    next        = nullptr;
    mathLib     = ClMathLib::instanceML();
}

//assignment, copy and move constructor and belonging methods

DenseOut& DenseOut::operator=(DenseOut other) {
    cout << "Assignment\n";
    std::swap(prev, other.prev);
    copyContent(other);
    return *this;
}


DenseOut::~DenseOut() {
  //  if(this->prev != nullptr)
  //      delete this->prev;
}

DenseOut::DenseOut(DenseOut &other) {
    if(other.prev != nullptr) {
        this -> prev = other.prev -> clone();
    }
    this->copyContent(other);
 }


DenseOut::DenseOut(DenseOut&& other) {
    prev = other.prev;
    this->copyContent(other);
    other.prev = nullptr;
 }

DenseOut* DenseOut::clone() {
    return new DenseOut(*this);
}

void DenseOut::copyContent(DenseOut& other) {
    this->errors         = other.errors;
    this->neurons        = other.neurons;
    //lossfunction   = other.lossfunction;
    this->activation     = other.activation;
    this->loss           = other.loss;
    this->mathLib        = other.mathLib;
    this->lossfunction   = other.lossfunction;
 }

void DenseOut::evalLoss(Activation activation) {
    switch(this -> loss) {
        case MEAN_SQUARED                       : switch(activation) {
                                                        case(SOFTMAX)   : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "smax_mean_squared" ); };
                                                                          break;
                                                        case(SIGMOID)   : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "sig_mean_squared"  ); };
                                                                          break;
                                                        case(TANH)      : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "tanh_mean_squared" ); };
                                                                          break;
                                                        case(RELU)      : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "relu_mean_squared" ); };
                                                                          break;
                                                  }
                                                  break;
        case CATEGORICAL_CROSS_ENTROPY          : switch(activation) {
                                                        case(SOFTMAX)   : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "smax_cat_crent"    ); };
                                                                          break;
                                                        case(SIGMOID)   : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "sig_cat_crent"     ); };
                                                                          break;
                                                        case(TANH)      : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "tanh_cat_crent"    ); };
                                                                          break;
                                                        case(RELU)      : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "relu_cat_crent"    ); };
                                                                          break;
                                                  }
                                                  break;
        case SPARE_CATEGORICAL_CROSS_ENTROPY    : switch(activation) {
                                                        case(SOFTMAX)   : lossfunction = [this](vector<float> &target) {lossSparseCrossSmax(target); };
                                                                          break;
                                                        case(SIGMOID)   : lossfunction = [this](vector<float> &target) {lossSparseCrossSig(target); };
                                                                          break;
                                                        case(TANH)      : lossfunction = [this](vector<float> &target) {lossSparseCrossTan(target); };
                                                                          break;
                                                        case(RELU)      : lossfunction = [this](vector<float> &target) {lossSparseCrossRel(target); };
                                                                          break;
                                                  }
                                                  break;
        case BINARY_CATEGORICAL_CROSS_ENTROPY   : switch(activation) {
                                                        case(SIGMOID)   : lossfunction = [this](vector<float> &target) {lossBinCrossSig(target); };
                                                                          break;
                                                        case(TANH)      : lossfunction = [this](vector<float> &target) {lossBinCrossTan(target); };
                                                                          break;
                                                        case(RELU)      : lossfunction = [this](vector<float> &target) {lossBinCrossRel(target); };
                                                                          break;
                                                        
                                                  }
                                                  break;
        case MULTI_CLASS_CROSS_ENTROPY          : switch(activation) {
                                                        case(SIGMOID)   : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "sig_mult_crent"     ); };
                                                                          break;
                                                        case(TANH)      : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "tanh_mult_crent" ); };
                                                                          break;
                                                        case(RELU)      : lossfunction = [this](vector<float> &target) { this->mathLib->vcErr(this->neurons, target, this->errors, "relu_mult_crent" ); };
                                                                          break;
                                                  };
                                                  break;
        default                                 : return;
    };
}
// since binary cross entropy should only be happening on one outputvector, we calculate the error on cpu

void DenseOut::lossBinCrossSig(vector<float> &target) {
    for(size_t i = 0; i < target.size(); i++) {
        this->errors[i] = neurons[i] - target[i];
    }
    
}


void DenseOut::lossBinCrossTan(vector<float> &target) {
    for(size_t i = 0; i < target.size(); i++) {
            this->errors[i] = ( ( neurons[i] - target[i] ) / ( NON_ZERO_CONSTANT_STAT + neurons[i] ) ) * ( 1 + neurons[i] );
    }
    
}


void DenseOut::lossBinCrossRel(vector<float> &target) {
    for(size_t i = 0; i < target.size(); i++) {
        if(neurons[i] > 0)
            this->errors[i] = ( ( neurons[i] - target[i] ) / (NON_ZERO_CONSTANT_STAT + ( neurons[i] * ( 1 - neurons[i] ) ) ) );
        else
            this->errors[i] = 0;
    }
    
}

void DenseOut::lossSparseCrossSmax(vector<float> &target) {
    
    size_t targidx = (size_t) target[0];
    for(size_t i = 0; i < neurons.size(); i++) {
        errors[i] = neurons[targidx] - (i == targidx ? 1 : 0);
    } 
}

void DenseOut::lossSparseCrossSig(vector<float> &target) {
    size_t siz = errors.size();
    errors.resize(0);
    errors.resize(siz);
    size_t targidx = (int) target[0];
    errors[targidx] = -( 1 - neurons[targidx] ) ;
   
}


void DenseOut::lossSparseCrossTan(vector<float> &target) {
    size_t siz = errors.size();
    errors.resize(0);
    errors.resize(siz);
    size_t targidx = (int) target[0];
    errors[targidx] = -(1/neurons[targidx] + NON_ZERO_CONSTANT_STAT) + neurons[targidx] ;    
}


void DenseOut::lossSparseCrossRel(vector<float> &target) {
   size_t siz = errors.size();
    errors.resize(0);
    errors.resize(siz);
    size_t targidx = (int) target[0];
    if(neurons[targidx] > 0 )
        errors[targidx] = -(1/( NON_ZERO_CONSTANT_STAT + neurons[targidx]));  
}

void DenseOut::fwd() {
    return;
}

void DenseOut::learn(const float learnRate) {
    return;
}

void DenseOut::bwd() {
    return;
 }

 void DenseOut::setupLayer() {
     if(this -> prev == nullptr) {
         cout << "Network not properly initialized\n";
         throw new std::logic_error("Structure missmatch");
     }
     if(( this -> loss == MULTI_CLASS_CROSS_ENTROPY ||  this -> loss  == BINARY_CATEGORICAL_CROSS_ENTROPY ) && this -> prev -> activation == SOFTMAX) {
        cout << "Useless combination of loss and activation, changing activation to sigmoid\n";
        this -> evalLoss(SIGMOID);
    } else {
        this -> evalLoss(this -> prev -> activation);
    }
 }
 
void DenseOut::closs(vector<float> &target) {
    this->lossfunction(target);
    this->prev->bwd();
}
#endif
