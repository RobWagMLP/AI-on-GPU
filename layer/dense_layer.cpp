#ifndef DENSE
#define DENSE

#include <dense_out.cpp>

class Dense: public Layer<float> {
    public:
        Dense(){};
        Dense(uint32_t outputSize, Activation iAct, Layer *iPrev, Layer *iNext, WeightInit iWeightInit);
        Dense(Activation iAct, Layer *iPrev, Layer *iNext, WeightInit iWeightInit);
        Dense(Dense & other);
        Dense(Dense &&other);
        ~Dense();

        Dense& operator=(Dense other);
        Dense* clone();

        void copyContent(Dense& other);
        void evalAct();

        void ctotalLoss(vector<float> &target, vector<float> &prediction);
        
        void fwd();
        void bwd();
        void learn(float learnRate);
        void closs(vector<float> &target);
        void setupWeights();

        void initAllRandom(const size_t &layerFrom, const size_t &layerTo);
        void initXavierUni(const size_t &layerFrom, const size_t &layerTo);
        void initXavierNorm(const size_t &layerFrom, const size_t &layerTo);
        void initGausian(const size_t &layerFrom, const size_t &layerTo);


        std::function<void()> activate;
        std::function<void()> activateDW;

        vector<float>   intermedErr;
        vector<float>   intermedDW;
        vector<float>   dWcollect;
        vector<float>   dwBiasCollect;

        WeightInit weightInit;
        bool       isInput;
    private:
        std::shared_ptr<ClMathLib> mathLib;
};

Dense::Dense(uint32_t outputSize, Activation iAct, Layer *iPrev, Layer *iNext, WeightInit iWeightInit = XAVIERNORMAL)
    :Layer<float>(outputSize)
    {
    activation  = iAct;
    prev        = iPrev; 
    next        = iNext;
    weightInit  = iWeightInit;
    mathLib     = ClMathLib::instanceML();
    this -> evalAct();
}

Dense::Dense(Activation iAct, Layer *iPrev, Layer *iNext, WeightInit iWeightInit = XAVIERNORMAL)
    :Layer<float>()
    {
    activation  = iAct;
    prev        = iPrev; 
    next        = iNext;
    weightInit  = iWeightInit;
    mathLib     = ClMathLib::instanceML();
    this -> evalAct();
}

//assignment, copy and move constructor and belonging methods

Dense& Dense::operator=(Dense other) {
    cout << "Assignment\n";
    std::swap(prev, other.prev);
    copyContent(other);
    return *this;
}


Dense::~Dense() {
    if(prev != nullptr)
        delete prev;
    if(next != nullptr)
        delete next;
}

Dense::Dense(Dense &other) {
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


Dense::Dense(Dense&& other) {
    prev = other.prev;
    next = other.next;
    copyContent(other);
    other.prev = nullptr;
    other.next = nullptr;
 }

Dense* Dense::clone() {
    return new Dense(*this);
}

void Dense::copyContent(Dense& other) {
    this->errors         = other.errors;
    this->neurons        = other.neurons;
    //lossfunction   = other.lossfunction;
    this->loss           = other.loss;
    this->weights        = other.weights;
    this->weight_width   = other.weight_width;
    this->bias           = other.bias;
    this->intermed       = other.intermed;
    this->intermedDW     = other.intermedDW;
    this->dW             = other.dW;
    this->dWcollect      = other.dWcollect;
    this->dwBiasCollect  = other.dwBiasCollect;
    this->weightInit     = other.weightInit;
    this->mathLib        = other.mathLib;
    this->intermedErr    = other.intermedErr;
    this->dWBias         = other.dWBias;
    evalAct();
 }

void Dense::evalAct() {
        switch(activation) {
            case(SIGMOID):  this->activate   = [this]() { this->mathLib->vcAct(this->intermed       , this->next->neurons, "vact_sig" ); };
                            this->activateDW = [this]() { this->mathLib->vcAct(this->neurons        , this->intermedDW   , "vact_sig_dw" ); };
                            break;
            case(RELU):     this->activate   = [this]() { this->mathLib->vcAct(this->intermed       , this->next->neurons, "vact_relu" ); };
                            this->activateDW = [this]() { this->mathLib->vcAct(this->neurons        , this->intermedDW   , "vact_relu_dw" ); };
                            break;
            case(TANH):     this->activate   = [this]() { this->mathLib->vcAct(this->intermed       , this->next->neurons, "vact_tanh" ); };
                            this->activateDW = [this]() { this->mathLib->vcAct(this->neurons        , this->intermedDW   , "vact_tanh_dw" ); };
                            break;
            case(SOFTMAX):  this->activate   = [this]() { this->mathLib->vcAct(this->intermed       , this->next->neurons, "vact_softmax" ); };
                            this->activateDW = [this]() { this->mathLib->vcAct(this->neurons        , this->intermedDW   , "vact_softmax_dw" ); };
                            break;
        default                                 : return;
    };
}

void Dense::fwd() {
    this->mathLib->mtPrdBias( this->weights, this->neurons, this->bias, this->intermed, this->intermed.size(), 1);
    this->activate();
    this->next->fwd();
}


void Dense::bwd() {
    //calc dWs

    this->mathLib->mtDyad(this->next->errors, this->neurons, this->dW);
    this->mathLib->vcAdd(this->dWcollect, this->dW, this->dWcollect);

    //calc dWs Bias

    this->mathLib->vcCpy(this->next->errors, this->dWBias);
    this->mathLib->vcAdd(this->dwBiasCollect, this->dWBias, this->dwBiasCollect);

     if(!this->isInput) {
        //calc error for this layer to use in prev layer
        this->mathLib->mtPrd(this->next->errors, this->weights, this->intermedErr, 1, this->weight_width);
        this->activateDW();
        this->mathLib->vcAdd(this->intermedErr, this->intermedDW, this->errors, "vmult");
        this->prev->bwd();
    }
}

void Dense::learn(const float learnRate) {
    this->mathLib->vcAddFct(this->weights,   this->dWcollect    , this->weights , learnRate );
    this->mathLib->vcAddFct(this->bias   ,   this->dwBiasCollect, this->bias    , learnRate );
    this->next->learn(learnRate);
}


void Dense::closs(vector<float> &target) {
    
}

void Dense::setupWeights() {
    this->isInput = this->prev == nullptr;
    const size_t layerFrom   = this ->         neurons.size();
    const size_t layerTo     = this -> next -> neurons.size(); 
    this->intermed           = vector<float>(layerTo);
    this->intermedDW         = vector<float>(layerFrom);
    this->intermedErr        = vector<float>(layerFrom);
    this->errors             = vector<float>(layerFrom);
    this->bias               = vector<float>(layerTo);
    this->dWBias             = vector<float>(layerTo);
    this->dwBiasCollect      = vector<float>(layerTo);
    this->weights            = vector<float>(layerTo*layerFrom);
    this->dWcollect          = vector<float>(layerTo*layerFrom);
    this->dW                 = vector<float>(layerTo*layerFrom);
    this->weight_width       = layerFrom;

    switch(this->weightInit) {
        case (ALLRANDOM):       this->initAllRandom(layerFrom, layerTo);
                                break;
        case (XAVIERNORMAL):    this->initXavierNorm(layerFrom, layerTo);
                                break;
        case (XAVIERUNIFORM):   this->initXavierUni(layerFrom, layerTo);
                                break;
        case (GAUSIAN):         this->initGausian(layerFrom, layerTo);
                                break;
        default:                return;
    }
}

void Dense::initAllRandom(const size_t &layerFrom, const size_t &layerTo) {
    for(size_t i = 0; i < layerTo; i++) {
        for(size_t j = 0; j < layerFrom; j++) {
            this->weights[layerFrom*i + j] = 2*((float)rand()/(float)RAND_MAX) - 1;
        }
        this->bias[i] = ((float)rand()/(float)RAND_MAX) - 1;
    }
}

void Dense::initXavierUni(const size_t &layerFrom, const size_t &layerTo) {
    const float limit = 1 / ( sqrt( float(layerFrom + 1) ) ); //+ 1 because of bias
    
    for(size_t i = 0; i < layerTo; i++) {
        for(size_t j = 0; j < layerFrom; j++) {
            this->weights[layerFrom*i + j] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
        }
        this->bias[i] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
    }
}

void Dense::initXavierNorm(const size_t &layerFrom, const size_t &layerTo) {
    const float limit = sqrt(6) / sqrt( ( (float) (layerFrom + layerTo + 1) ) ); //+ 1 because of bias
    
    for(size_t i = 0; i < layerTo; i++) {
        for(size_t j = 0; j < layerFrom; j++) {
            this->weights[layerFrom*i + j] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
        }
        this->bias[i] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
    }
}
//mostly for relu activation
void Dense::initGausian(const size_t &layerFrom, const size_t &layerTo) {
    const float limit = sqrt(2) / ( layerFrom + 1 ) ; //+ 1 because of bias
    
    for(size_t i = 0; i < layerTo; i++) {
        for(size_t j = 0; j < layerFrom; j++) {
            this->weights[layerFrom*i + j] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
        }
        this->bias[i] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
    }
}



#endif