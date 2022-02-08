#include <basic_layer.cpp>

class Dense: public Layer {
    public:
        Dense(){};
        Dense(uint32_t outputSize,Loss iLoss, Activation iAct, Layer *iPrev, Layer *iNext);
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
        void learn();
        void closs(vector<float> &target);
        void setupWeights();

        std::function<void(vector<float>&)> activate;
        std::function<void(vector<float>&)> dAct;
    private:
        std::shared_ptr<ClMathLib> mathLib;
};

Dense::Dense(uint32_t outputSize, Loss iLoss, Activation iAct, Layer *iPrev, Layer *iNext)
    :Layer(outputSize)
    {
    loss        = iLoss;
    activation  = iAct;
    prev        = iPrev; 
    next        = iNext;
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
    this->activation     = other.activation;
    this->loss           = other.loss;
    this->weights        = other.weights;
    this->weight_width   = other.weight_width;
    this->bias           = other.bias;
    this->intermed       = other.intermed;
    this->tot_errs       = other.tot_errs;
    this->dW             = other.dW;
    this->mathLib  = other.mathLib;
    evalAct();
 }

void Dense::evalAct() {
        switch(activation) {
            case(SOFTMAX):
                            break;
            case(RELU):
                            break;
            case(TANH):
                            break;
            case(SIGMOID):
                            break;
        default                                 : return;
    };
}

void Dense::fwd() {
}


void Dense::bwd() {

}

void Dense::learn() {
    return;
}


void Dense::closs(vector<float> &target) {
    
}

void Dense::setupWeights() {
    const size_t layerFrom   = this ->         neurons.size();
    const size_t layerTo     = this -> next -> neurons.size(); 
}




