#ifndef CONV2D
#define CONV2D

#include "basic_layer.cpp"

class Conv2D: public Layer {
    public:
        Conv2D(){ this -> type = ConvolutionalLayer; };
        Conv2D(vector<size_t> inpDims, std::array<size_t, 2> kernelDims, size_t convolutions, Activation iAct, WeightInit iWeightInit);
        Conv2D(size_t convolutions, Activation iAct, WeightInit iWeightInit);
        Conv2D(Conv2D & other);
        Conv2D(Conv2D &&other);
        ~Conv2D();

        Conv2D& operator=(Conv2D other);
        Conv2D* clone();

        void copyContent(Conv2D& other);
        void evalActDw(Activation activationPrev);
        void evalAct();

        void ctotalLoss(vector<float> &target, vector<float> &prediction);
        
        void fwd();
        void bwd();
        void learn(const float learnRate);
        void closs(vector<float> &target);
        void setupLayer();

        void initAllRandom(const size_t &channels, const size_t &kernelX, const size_t &kernelY);
        void initXavierUni(const size_t &channels, const size_t &kernelX, const size_t &kernelY);
        void initXavierNorm(const size_t &channels, const size_t &kernelX, const size_t &kernelY);
        void initGausian(const size_t &channels, const size_t &kernelX, const size_t &kernelY);


        std::function<void()> activate;
        std::function<void()> activateDW;

        vector<float>   intermedErr;
        vector<float>   dWcollect;
        vector<float>   dwBiasCollect;

        WeightInit weightInit;
        bool       isInput;
        size_t     convolutions;
        std::array<size_t, 2> kernelDims;
        vector<size_t> inpDims;
        vector<size_t> outDims;


    private:
        std::shared_ptr<ClMathLib> mathLib;
};

Conv2D::Conv2D(vector<size_t> inpDims, std::array<size_t, 2> kernelDims, size_t convolutions, Activation iAct, WeightInit iWeightInit)
    :Layer()
    {
    this -> activation   = iAct;
    this -> weightInit   = iWeightInit;
    this -> mathLib      = ClMathLib::instanceML();
    this -> convolutions = convolutions;
    this -> kernelDims   = kernelDims;
    this -> inpDims      = inpDims;
    this -> type         = ConvolutionalLayer;
    this -> evalAct();
}

Conv2D::Conv2D(size_t convolutions, Activation iAct, WeightInit iWeightInit)
    :Layer()
    {
    this -> convolutions = convolutions;
    this -> activation   = iAct;
    this -> weightInit   = iWeightInit;
    this -> mathLib      = ClMathLib::instanceML();
    this -> kernelDims   = {3, 3};
    this -> inpDims      = inpDims;
    this -> type         = ConvolutionalLayer;
    this -> evalAct();
}

//assignment, copy and move constructor and belonging methods

Conv2D& Conv2D::operator=(Conv2D other) {
    cout << "Assignment\n";
    std::swap(this->prev, other.prev);
    std::swap(this->next, other.next);
    this->copyContent(other);
    return *this;
}


Conv2D::~Conv2D() {
   /* if(prev != nullptr)
        delete prev;
    if(next != nullptr)
        delete next;*/
}

Conv2D::Conv2D(Conv2D &other) {
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


Conv2D::Conv2D(Conv2D&& other) {
    this->prev = other.prev;
    this->next = other.next;
    this->copyContent(other);
    other.prev = nullptr;
    other.next = nullptr;
 }

Conv2D* Conv2D::clone() {
    return new Conv2D(*this);
}

void Conv2D::copyContent(Conv2D& other) {
    this->errors         = other.errors;
    this->neurons        = other.neurons;
    //lossfunction   = other.lossfunction;
    this->loss           = other.loss;
    this->activation     = other.activation;
    this->weights        = other.weights;
    this->weight_width   = other.weight_width;
    this->bias           = other.bias;
    this->intermed       = other.intermed;
    this->dWcollect      = other.dWcollect;
    this->dwBiasCollect  = other.dwBiasCollect;
    this->weightInit     = other.weightInit;
    this->mathLib        = other.mathLib;
    this->intermedErr    = other.intermedErr;
    this -> type         = ConvolutionalLayer;
    if(other.prev != nullptr) {
        this -> evalActDw(other.prev -> activation);
    }
 }

void Conv2D::evalAct() {
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

void Conv2D::evalActDw(Activation activationPrev) {
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

void Conv2D::setupLayer() {
    this->isInput = this->prev == nullptr;
    if( this->next == nullptr ) {
         cout << "Network not initializied properly \n";
         throw new std::logic_error("Structure missmatch");
    }
    if( this -> inpDims.size() < 2 || this -> inpDims.size() > 3) {
        cout << "Input Dimensions don't fit 2D Convolution \n";
        throw new std::logic_error("Structure missmatch");
    }
    size_t layerFrom = 1;
    for(size_t i = 0; i < this -> inpDims.size(); i++) {
        layerFrom *= inpDims[i];
    }
    const uint8_t newDimsX = this -> inpDims[0] - (this -> kernelDims[0] - 1);
    const uint8_t newDimsY = this -> inpDims[1] - (this -> kernelDims[1] - 1);

    this -> outDims = {newDimsX, newDimsY, convolutions};

    if( this -> next -> type == ConvolutionalLayer ||  this -> next -> type == PoolingLayer )
        this -> next -> inpDims = { newDimsX, newDimsY, this -> convolutions};
    else
        this -> next -> neurons = vector<float>( newDimsX * newDimsY * this ->convolutions );
    if(this->inpDims.size() <= 2) {
        this -> inpDims.push_back(1); //either we have a 2-dimensional input with only one channel or we have the channel in the third dimension.
    }
    const size_t layerTo      = newDimsX * newDimsY * this -> convolutions;
    const size_t channels     = this -> inpDims[2]; 
    const size_t kernels      = channels * this -> convolutions;
    const size_t weightLength = kernels * inpDims[0] * inpDims[1];

    this -> neurons         = vector<float>(layerFrom);
    this -> intermed        = vector<float>(layerFrom * this -> convolutions );

    this -> errors          = vector<float>(layerFrom);
    this -> intermedErr     = vector<float>(layerFrom);

    this -> weights         = vector<float>(weightLength);
    this -> dWcollect       = vector<float>(weightLength);

    this -> bias            = vector<float>(convolutions); //ToDO: add option for tied bias
    this -> dwBiasCollect   = vector<float>(convolutions);

    if(this -> prev != nullptr) {
        this -> evalActDw(this -> prev -> activation);
    }

    switch(this->weightInit) {
        case (ALLRANDOM):       this->initAllRandom(channels, kernelDims[0], kernelDims[1]);
                                break;
        case (XAVIERNORMAL):    this->initXavierNorm(channels, kernelDims[0], kernelDims[1]);
                                break;
        case (XAVIERUNIFORM):   this->initXavierUni(channels, kernelDims[0], kernelDims[1]);
                                break;
        case (GAUSIAN):         this->initGausian(channels, kernelDims[0], kernelDims[1]);
                                break;
        default:                return;
    }
    this -> next -> setupLayer();
}

void Conv2D::initAllRandom(const size_t &channels, const size_t &kernelX, const size_t &kernelY) {
    for(size_t i = 0; i < ( channels*this -> convolutions ); i++) {
        for(size_t j = 0; j < kernelY; j++) {
            for(size_t k = 0; k < kernelX; k++) {
                this->weights[i*kernelX*kernelY + j*kernelX + k] = 2*((float)rand()/(float)RAND_MAX) - 1;
            }
        }
        if(i < this-> convolutions) {
            this->bias[i] = ((float)rand()/(float)RAND_MAX) - 1;
        }
    }
}

void Conv2D::initXavierUni(const size_t &channels, const size_t &kernelX, const size_t &kernelY) {
    const float limit = 1 / ( sqrt( float(channels * kernelDims[0] * kernelDims[1] + 1) ) ); //+ 1 because of bias
    
    for(size_t i = 0; i < ( channels*this -> convolutions ); i++) {
        for(size_t j = 0; j < kernelY; j++) {
            for(size_t k = 0; k < kernelX; k++) {
                this->weights[i*kernelX*kernelY + j*kernelX + k] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
            }
        }
        if(i < this-> convolutions) {
            this->bias[i] = ((float)rand()/(float)RAND_MAX) - 1;
        }
    }
}

void Conv2D::initXavierNorm(const size_t &channels, const size_t &kernelX, const size_t &kernelY) {
    const float limit = sqrt(6) / sqrt( ( (float) ( ( channels * kernelDims[0] * kernelDims[1] ) + ( convolutions * kernelDims[0] * kernelDims[1] ) + 1) ) ); //+ 1 because of bias
    
    for(size_t i = 0; i < ( channels*this -> convolutions ); i++) {
        for(size_t j = 0; j < kernelY; j++) {
            for(size_t k = 0; k < kernelX; k++) {
                this->weights[i*kernelX*kernelY + j*kernelX + k] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
            }
        }
        if(i < this-> convolutions) {
            this->bias[i] = ((float)rand()/(float)RAND_MAX) - 1;
        }
    }
}
//mostly for relu activation
void Conv2D::initGausian(const size_t &channels, const size_t &kernelX, const size_t &kernelY) {
    const float limit = sqrt(2) / ( ( ( channels * kernelDims[0] * kernelDims[1] ) ) + 1 ) ; //+ 1 because of bias
    
    for(size_t i = 0; i < ( channels*this -> convolutions ); i++) {
        for(size_t j = 0; j < kernelY; j++) {
            for(size_t k = 0; k < kernelX; k++) {
                this->weights[i*kernelX*kernelY + j*kernelX + k] = 2*limit*((float)rand()/(float)RAND_MAX) - limit;
            }
        }
        if(i < this-> convolutions) {
            this->bias[i] = ((float)rand()/(float)RAND_MAX) - 1;
        }
    }
}


#endif