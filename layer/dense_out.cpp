#ifndef DENSEOUT
#define DENSEOUT

#include <basic_layer.cpp>
#include <mutex>
#include <thread>

constexpr float NON_ZERO_CONSTANT = 0.000001;

class DenseOut: public Layer<float> {
    public:
        DenseOut(){};
        DenseOut(uint32_t outputSize,Loss iLoss, Activation iAct, Layer *iPrev);
        DenseOut(DenseOut & other);
        DenseOut(DenseOut &&other);
        ~DenseOut();

        DenseOut& operator=(DenseOut other);
        DenseOut* clone();

        void copyContent(DenseOut& other);
        void evalLoss();
        void lossBinCrossSig(vector<float> &target);
        void lossBinCrossTan(vector<float> &target);
        void lossBinCrossRel(vector<float> &target);

        void lossSparseCrossSmax(vector<float> &target);
        void lossSparseCrossSig(vector<float> &target);
        void lossSparseCrossTan(vector<float> &target);
        void lossSparseCrossRel(vector<float> &target);

        void totalLossMeanSq(vector<float> &target, vector<float> &acts);
        void totalLossbinCross(vector<float> &target, vector<float> &acts);
        void totalLossCrEnt(vector<float> &target, vector<float> &acts);
        void totalLossSprsCrEnt(vector<float> &target, vector<float> &acts);

        void ctotalLoss(vector<float> &target, vector<float> &prediction);
        
        void fwd();
        void bwd();
        void learn(float learnRate);
        void closs(vector<float> &target);

        std::function<void(vector<float>&)> lossfunction;
        vector<float> tot_errs;
    private:
        std::shared_ptr<ClMathLib> mathLib;
};

DenseOut::DenseOut(uint32_t outputSize, Loss iLoss, Activation iAct, Layer *iPrev)
    :Layer<float>(outputSize),
    tot_errs(0) {
    loss        = iLoss;
    activation  = iAct;
    prev        = iPrev; 
    next        = nullptr;
    mathLib     = ClMathLib::instanceML();

    if(( iLoss == MULTI_CLASS_CROSS_ENTROPY || iLoss == BINARY_CATEGORICAL_CROSS_ENTROPY) && iAct == SOFTMAX) {
        cout << "Useless combination of loss and activation, changing activation to sigmoid\n";
        activation = SIGMOID;
    }
    this -> evalLoss();
}

//assignment, copy and move constructor and belonging methods

DenseOut& DenseOut::operator=(DenseOut other) {
    cout << "Assignment\n";
    std::swap(prev, other.prev);
    copyContent(other);
    return *this;
}


DenseOut::~DenseOut() {
    if(prev != nullptr)
        delete prev;
}

DenseOut::DenseOut(DenseOut &other) {
    //std::allocator<Layer> a;
    //prev = a.allocate(sizeof(*prev));
    if(other.prev != nullptr) {
        this -> prev = other.prev -> clone();
    }
    copyContent(other);
 }


DenseOut::DenseOut(DenseOut&& other) {
    prev = other.prev;
    copyContent(other);
    other.prev = nullptr;
 }

DenseOut* DenseOut::clone() {
    return new DenseOut(*this);
}

void DenseOut::copyContent(DenseOut& other) {
    errors         = other.errors;
    neurons        = other.neurons;
    //lossfunction   = other.lossfunction;
    activation     = other.activation;
    loss           = other.loss;
    tot_errs       = other.tot_errs;
    this->mathLib  = other.mathLib;
    evalLoss();
 }

void DenseOut::evalLoss() {
    switch(loss) {
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
                                                        case(SIGMOID)   : lossfunction = [this](vector<float> &target) {lossBinCrossSig(target); };
                                                                          break;
                                                        case(TANH)      : lossfunction = [this](vector<float> &target) {lossBinCrossTan(target); };
                                                                          break;
                                                        case(RELU)      : lossfunction = [this](vector<float> &target) {lossBinCrossRel(target); };
                                                                          break;
                                                  }
                                                  break;
        case BINARY_CATEGORICAL_CROSS_ENTROPY   : switch(activation) {
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
        errors[i] = neurons[i] - target[i];
    }
    
}


void DenseOut::lossBinCrossTan(vector<float> &target) {
    for(size_t i = 0; i < target.size(); i++) {
            errors[i] = ( ( neurons[i] - target[i] ) / ( NON_ZERO_CONSTANT + neurons[i] ) ) * ( 1 + neurons[i] );
    }
    
}


void DenseOut::lossBinCrossRel(vector<float> &target) {
    for(size_t i = 0; i < target.size(); i++) {
        if(neurons[i] > 0)
            errors[i] = ( ( neurons[i] - target[i] ) / (NON_ZERO_CONSTANT + ( neurons[i] * ( 1 - neurons[i] ) ) ) );
        else
            errors[i] = 0;
    }
    
}

void DenseOut::lossSparseCrossSmax(vector<float> &target) {
    size_t siz = errors.size();
    errors.resize(0);
    errors.resize(siz);
    size_t targidx = (size_t) target[0];
    errors[targidx] = neurons[targidx] - 1;    
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
    errors[targidx] = -(1/neurons[targidx] + NON_ZERO_CONSTANT) + neurons[targidx] ;    
}


void DenseOut::lossSparseCrossRel(vector<float> &target) {
   size_t siz = errors.size();
    errors.resize(0);
    errors.resize(siz);
    size_t targidx = (int) target[0];
    if(neurons[targidx] > 0 )
        errors[targidx] = -(1/( NON_ZERO_CONSTANT + neurons[targidx]));  
}

void DenseOut::totalLossMeanSq(vector<float> &target, vector<float> &acts ) {
    float error = 0.;
    for(size_t i = 0; i < acts.size(); i++) {
        error += pow( (acts[i] - target[i] ), 2);
    }
    this->tot_errs.push_back(error/(float)acts.size() );
}
void DenseOut::totalLossbinCross(vector<float> &target, vector<float> &acts ) {
    float error = 0;
    for(size_t i = 0; i < acts.size(); i++) {
        error -= target[i] * log( NON_ZERO_CONSTANT + acts[i] ) + ( 1 - target[i])*log( NON_ZERO_CONSTANT + 1 - acts[i]);
    }
    this->tot_errs.push_back(error/(float)acts.size() );
}
void DenseOut::totalLossCrEnt(vector<float> &target, vector<float> &acts ) {
    float error = 0;
    for(size_t i = 0; i < acts.size(); i++) {
        error -= target[i] * log( acts[i] + NON_ZERO_CONSTANT);
    }
    this->tot_errs.push_back(error/(float)acts.size() );
}

void DenseOut::totalLossSprsCrEnt(vector<float> &target, vector<float> &acts ) {
    tot_errs.push_back( -log( acts[target[0]] ) );
    cout << "Thread part\n";
}

void DenseOut::ctotalLoss(vector<float> &target, vector<float> &prediction) {
    switch(this->loss) {
        case(MEAN_SQUARED):                    this->totalLossMeanSq(   target, prediction);
                                               break;
        case(BINARY_CATEGORICAL_CROSS_ENTROPY):
        case(MULTI_CLASS_CROSS_ENTROPY):       this->totalLossbinCross( target, prediction);
                                               break;
        case(CATEGORICAL_CROSS_ENTROPY):       this->totalLossCrEnt(    target, prediction);
                                               break;
        case(SPARE_CATEGORICAL_CROSS_ENTROPY): this->totalLossSprsCrEnt(target, prediction);
                                               break;   
    }
}


void DenseOut::fwd() {
}


void DenseOut::bwd() {

}

void DenseOut::learn(float learnRate) {
    return;
}


void DenseOut::closs(vector<float> &target) {
    this->lossfunction(target);
    std::thread t([this, &target] { this->ctotalLoss(target, this->neurons); } );
    this->prev->bwd();
    t.join();
}




#endif
