#ifndef DENSE
#define DENSE

#include "basic_layer.cpp"

class Dense: public Layer {
    public:
        Dense(){ this -> type = DenseLayer; };
        Dense(uint32_t outputSize, Activation iAct, WeightInit iWeightInit,Optimization optimizer);
        Dense(Activation iAct, WeightInit iWeightInit, Optimization optimizer);
        Dense(Dense & other);
        Dense(Dense &&other);
        ~Dense();

        Dense& operator=(Dense other);
        shared_ptr<Layer> clone();
        void fwd();
        void bwd();
        void learn(const float learnRate);
        void closs(vector<float> &target);
        void setupLayer();
        void setInput(vector<float> & inp);
  
    private:
        void copyContent(Dense& other);
        void swap(Dense& other);
        void evalActDw(Activation activationPrev);
        void evalAct();
        void evalOptimizer();
        
        void initAllRandom(const size_t &layerFrom, const size_t &layerTo);
        void initXavierUni(const size_t &layerFrom, const size_t &layerTo);
        void initXavierNorm(const size_t &layerFrom, const size_t &layerTo);
        void initGausian(const size_t &layerFrom, const size_t &layerTo);


        std::function<void()> activate;
        std::function<void()> activateDW;

        vector<float>   intermedErr;
        vector<float>   dWcollect;
        vector<float>   dwBiasCollect;

        WeightInit weightInit;
        bool       isInput;

    private:
        std::shared_ptr<ClMathLib> mathLib;
};

Dense::Dense(uint32_t outputSize, Activation iAct, WeightInit iWeightInit = XAVIERNORMAL, Optimization optimizer = SGD)
    :Layer(outputSize)
    {
    this -> activation   = iAct;
    this -> prev         = nullptr; 
    this -> next         = nullptr;
    this -> weightInit   = iWeightInit;
    this -> mathLib      = ClMathLib::instanceML();
    this -> type         = DenseLayer;
    this -> optimizer    = optimizer;
    this->evalAct();
    this -> evalOptimizer();
}

Dense::Dense(Activation iAct, WeightInit iWeightInit = XAVIERNORMAL, Optimization optimizer = SGD)
    :Layer()
    {
    this -> activation   = iAct;
    this -> prev         = nullptr; 
    this -> next         = nullptr;
    this -> weightInit   = iWeightInit;
    this -> mathLib      = ClMathLib::instanceML();
    this -> type         = DenseLayer;
    this -> optimizer    = optimizer;
    this -> evalAct();
    this -> evalOptimizer();
}

//assignment, copy and move constructor and belonging methods

Dense& Dense::operator=(Dense other) {
    this -> swap(other);
    return *this;
}


Dense::~Dense() {
   /* if(prev != nullptr)
        delete prev;
    if(next != nullptr)
        delete next;*/
}

Dense::Dense(Dense &other): Layer() {
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


Dense::Dense(Dense&& other): Layer() {
    this -> swap(other);
 }
shared_ptr<Layer> Dense::clone() {
    return shared_ptr<Dense> (new Dense(*this));
}

void Dense::copyContent(Dense& other) {
    this -> errors         = other.errors;
    this -> neurons        = other.neurons;
    //lossfunction   = other.lossfunction;
    this -> loss           = other.loss;
    this -> activation     = other.activation;
    this -> weights        = other.weights;
    this -> weight_width   = other.weight_width;
    this -> bias           = other.bias;
    this -> intermed       = other.intermed;
    this -> dWcollect      = other.dWcollect;
    this -> dwBiasCollect  = other.dwBiasCollect;
    this -> weightInit     = other.weightInit;
    this -> mathLib        = other.mathLib;
    this -> movAvg         = other.movAvg;
    this -> movExp         = other.movExp;
    this -> movAvgB        = other.movAvgB;
    this -> movExpB        = other.movExpB;
    this -> intermedErr    = other.intermedErr;
    this -> optimizer      = other.optimizer;
    this -> type           = DenseLayer;
    if(other.prev != nullptr) {
        this -> evalActDw(other.prev -> activation);
    }
    this -> evalOptimizer();
 }
 void Dense::swap(Dense& other) {
    using std::swap;
    swap(this -> errors       , other.errors);
    swap(this -> neurons      , other.neurons);
    swap(this -> prev         , other.prev);
    swap(this -> bias         , other.bias);
    swap(this -> intermed     , other.intermed);
    swap(this -> dWcollect    , other.dWcollect);
    swap(this -> dwBiasCollect, other.dwBiasCollect);
    swap(this -> intermedErr  , other.intermedErr);
    swap(this -> movAvg       , other.movAvg);
    swap(this -> movAvgB      , other.movAvgB);
    swap(this -> movExpB      , other.movExpB);
    swap(this -> movExp       , other.movExp);
    swap(this -> next         , other.next);
    swap(this -> prev         , other.prev);
    //lossfunction   = other.lossfunction;
    this -> activation     = other.activation;
    this -> mathLib        = other.mathLib;
    this -> type           = OutputLayer;
    this -> optimizer      = other.optimizer;
    this -> weight_width   = other.weight_width;
    this -> weightInit     = other.weightInit;
    this -> type           = DenseLayer;
    if(other.prev != nullptr) {
        this -> evalActDw(other.prev -> activation);
    }
    this -> evalOptimizer();
 }

void Dense::evalAct() {
    switch(this -> activation) {
            case(SIGMOID):  this->activate   = [this]() { this->mathLib->vcAct(this->intermed       , this->next->neurons, "vact_sig" ); };
                            break;
            case(RELU):     this->activate   = [this]() { this->mathLib->vcAct(this->intermed       , this->next->neurons, "vact_relu" ); };
                            break;
            case(TANH):     this->activate   = [this]() { this->mathLib->vcAct(this->intermed       , this->next->neurons, "vact_tanh" ); };
                            break;
            case(SOFTMAX):  this->activate   = [this]() { this->mathLib->vcAct(this->intermed       , this->next->neurons, "vact_softmax" ); };
                            break;
        default                                 : return;
    };
}

void Dense::evalActDw(Activation activationPrev) {
    switch(activationPrev) {
            case(SIGMOID):  this->activateDW = [this]() { this->mathLib->vcDwMt(this->neurons        , this->intermedErr, this->errors   , "vact_sig_dw" ); };
                            break;
            case(RELU):     this->activateDW = [this]() { this->mathLib->vcDwMt(this->neurons        , this->intermedErr, this->errors  , "vact_relu_dw" ); };
                            break;
            case(TANH):     this->activateDW = [this]() { this->mathLib->vcDwMt(this->neurons        , this->intermedErr, this->errors   , "vact_tanh_dw" ); };
                            break;
            case(SOFTMAX):  this->activateDW = [this]() { this->mathLib->vcDwMt(this->neurons        , this->intermedErr, this->errors   , "vact_softmax_dw" ); };
                            break;
        default                                 : return;
    };
}

void Dense::evalOptimizer() {
    switch(this -> optimizer) {
            case(SGD):      this->optimize = [this](float learnRate) { this->mathLib->vcAddFct(this->weights,   this->dWcollect    , learnRate );
                                                                       this->mathLib->vcAddFct(this->bias   ,   this->dwBiasCollect, learnRate ); };
                            break;
            case(ADAM):     this->optimize = [this](float learnRate) { this -> mathLib -> vcAddFctAdam(this -> weights, this -> dWcollect    , this -> movAvg , this -> movExp , learnRate, this -> run);
                                                                       this -> mathLib -> vcAddFctAdam(this -> bias   , this -> dwBiasCollect, this -> movAvgB, this -> movExpB, learnRate, this -> run); };
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
    this->mathLib->mtDyad(this->next->errors, this->neurons, this->dWcollect);
    //calc dWs Bias

    this->mathLib->vcAdd(this->dwBiasCollect, this -> next -> errors, this->dwBiasCollect);

     if(!this->isInput) {
        //calc error for this layer to use in prev layer
        this -> mathLib -> mtPrd(this->next->errors, this->weights, this->intermedErr, 1, this->weight_width);
        this -> activateDW();
        this -> prev -> bwd();
    }
}

void Dense::learn(const float learnRate) {
    this -> optimize(learnRate);
    ++ this -> run;
    this->next->learn(learnRate);
}


void Dense::closs(vector<float> &target) {
    return;
}

void Dense::setInput(vector<float> & inp) {
    if( inp.size() != this -> neurons.size() ) {
        cout << "Cant't start Training. Input doesnt match Network structure \n";
        throw new std::logic_error("Structure missmatch");
    }
    this -> neurons = inp;
};

void Dense::setupLayer() {
    this->isInput = this->prev == nullptr;
    if( this->next == nullptr ) {
         cout << "Network not initializied properly \n";
         throw new std::logic_error("Structure missmatch");
    }
    const size_t layerFrom   = this ->         neurons.size();
    const size_t layerTo     = this -> next -> neurons.size(); 

    this->intermed           = vector<float>(layerTo);
    this->intermedErr        = vector<float>(layerFrom);
    this->errors             = vector<float>(layerFrom);
    this->bias               = vector<float>(layerTo);
    this->dwBiasCollect      = vector<float>(layerTo);
    this->weights            = vector<float>(layerTo*layerFrom);
    this->dWcollect          = vector<float>(layerTo*layerFrom);

    this -> movAvg           = vector<float>(layerTo*layerFrom);
    this -> movExp           = vector<float>(layerTo*layerFrom);
    this -> movAvgB          = vector<float>(layerTo);
    this -> movExpB          = vector<float>(layerTo);
    this->weight_width       = layerFrom;
    
    if(this -> prev != nullptr) {
        this -> evalActDw(this -> prev -> activation);
    }

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
    this -> next -> setupLayer();
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