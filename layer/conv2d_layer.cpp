#ifndef CONV2D
#define CONV2D

#include "basic_layer.cpp"

class Conv2D: public Layer {
    public:
        Conv2D(){ this -> type = ConvolutionalLayer; };
        Conv2D(array<size_t, 3> inpDims, std::array<size_t, 2> kernelDims, size_t convolutions, Activation iAct, WeightInit iWeightInit, Optimization optimizer);
        Conv2D(std::array<size_t, 2> kernelDims, size_t convolutions, Activation iAct, WeightInit iWeightInit, Optimization optimizer);
        Conv2D(Conv2D & other);
        Conv2D(Conv2D &&other);
        ~Conv2D();

        Conv2D& operator=(Conv2D other);
        shared_ptr<Layer> clone();

        void fwd();
        void bwd();
        void learn(const float learnRate);
        void closs(vector<float> &target);
        void setupLayer();
        void setInput(vector<float> & inp);
        void summary();

    private:
        void copyContent(Conv2D& other);
        void swap(Conv2D& other);
        void evalActDw(Activation activationPrev);
        void evalAct();
        void evalOptimizer();


        void initAllRandom(const size_t &channels, const size_t &kernelX, const size_t &kernelY);
        void initXavierUni(const size_t &channels, const size_t &kernelX, const size_t &kernelY);
        void initXavierNorm(const size_t &channels, const size_t &kernelX, const size_t &kernelY);
        void initGausian(const size_t &channels, const size_t &kernelX, const size_t &kernelY);


        std::function<void()> activate;
        std::function<void()> activateDW;

        vector<float>   intermedErr;
        vector<float>   dWcollect;
        vector<float>   dwBiasCollect;
        vector<float>   summFeatureMaps;
        vector<float>   summErrors;

        WeightInit weightInit;
        bool       isInput;
        size_t     convolutions;
        std::array<size_t, 2> kernelDims;
        array<size_t, 3> outDims;


    private:
        std::shared_ptr<ClMathLib> mathLib;
};

Conv2D::Conv2D(array<size_t, 3>inpDims, std::array<size_t, 2> kernelDims, size_t convolutions, Activation iAct, WeightInit iWeightInit, Optimization optimizer = SGD)
    :Layer()
    {
    this -> activation   = iAct;
    this -> weightInit   = iWeightInit;
    this -> mathLib      = ClMathLib::instanceML();
    this -> convolutions = convolutions;
    this -> kernelDims   = kernelDims;
    this -> inpDims      = inpDims;
    this -> type         = ConvolutionalLayer;
    this -> optimizer    = optimizer;
    this -> evalAct();
    this -> evalOptimizer();
}

Conv2D::Conv2D(std::array<size_t, 2> kernelDims, size_t convolutions, Activation iAct, WeightInit iWeightInit, Optimization optimizer = SGD)
    :Layer()
    {
    this -> convolutions = convolutions;
    this -> activation   = iAct;
    this -> weightInit   = iWeightInit;
    this -> mathLib      = ClMathLib::instanceML();
    this -> kernelDims   = kernelDims;
    this -> type         = ConvolutionalLayer;
    this -> inpDims      = array<size_t, 3> ();
    this -> optimizer    = optimizer;
    this -> evalAct();
    this -> evalOptimizer();
}

//assignment, copy and move constructor and belonging methods

Conv2D& Conv2D::operator=(Conv2D other) {
    this -> swap(other);
    return *this;
}


Conv2D::~Conv2D() {
   /* if(prev != nullptr)
        delete prev;
    if(next != nullptr)
        delete next;*/
}

Conv2D::Conv2D(Conv2D &other): Layer() {
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


Conv2D::Conv2D(Conv2D&& other): Layer() {
    this -> swap(other);
 }

shared_ptr<Layer> Conv2D::clone() {
    return shared_ptr<Conv2D>(new Conv2D(*this));
}

void Conv2D::copyContent(Conv2D& other) {
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
    this -> dWcollect      = other.dWcollect;
    this -> dwBiasCollect  = other.dwBiasCollect;
    this -> weightInit     = other.weightInit;
    this -> movAvg         = other.movAvg;
    this -> movExp         = other.movExp;
    this -> movAvgB        = other.movAvgB;
    this -> movExpB        = other.movExpB;
    this -> mathLib        = other.mathLib;
    this -> intermedErr    = other.intermedErr;
    this -> type           = ConvolutionalLayer;
    this -> optimizer      = other.optimizer;
    if(other.prev != nullptr) {
        this -> evalActDw(other.prev -> activation);
    }
    this -> evalOptimizer();
 }

void Conv2D::swap(Conv2D& other) {
    using std::swap;
    swap(this -> errors       , other.errors);
    swap(this -> neurons      , other.neurons);
    swap(this -> prev         , other.prev);
    swap(this -> bias         , other.bias);
    swap(this -> intermed     , other.intermed);
    swap(this -> dWcollect    , other.dWcollect);
    swap(this -> dwBiasCollect, other.dwBiasCollect);
    swap(this -> inpDims      , other.inpDims);
    swap(this -> outDims      , other.outDims);
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
 
void Conv2D::evalAct() {
    switch(this -> activation) {
            case(SIGMOID):  this->activate   = [this]() { this->mathLib->vcAct(this->summFeatureMaps       , this->next->neurons, "vact_sig" ); };
                            break;
            case(RELU):     this->activate   = [this]() { this->mathLib->vcAct(this->summFeatureMaps       , this->next->neurons, "vact_relu" ); };
                            break;
            case(TANH):     this->activate   = [this]() { this->mathLib->vcAct(this->summFeatureMaps       , this->next->neurons, "vact_tanh" ); };
                            break;
            case(SOFTMAX):  this->activate   = [this]() { this->mathLib->vcAct(this->summFeatureMaps       , this->next->neurons, "vact_softmax" ); };
                            break;
        default                                 : return;
    };
}

void Conv2D::evalActDw(Activation activationPrev) {
    switch(activationPrev) {
            case(SIGMOID):  this->activateDW = [this]() { this->mathLib->vcDwMt(this -> neurons        , this -> summErrors, this->errors   , "vact_sig_dw" ); };
                            break;
            case(RELU):     this->activateDW = [this]() { this->mathLib->vcDwMt(this -> neurons        , this -> summErrors, this->errors  , "vact_relu_dw" ); };
                            break;
            case(TANH):     this->activateDW = [this]() { this->mathLib->vcDwMt(this -> neurons        , this -> summErrors, this->errors   , "vact_tanh_dw" ); };
                            break;
            case(SOFTMAX):  this->activateDW = [this]() { this->mathLib->vcDwMt(this -> neurons        , this -> summErrors, this->errors   , "vact_softmax_dw" ); };
                            break;
        default                                 : return;
    };
}

void Conv2D::evalOptimizer() {
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



void Conv2D::setupLayer() {
    this->isInput = this->prev == nullptr;
    if( this->next == nullptr ) {
         cout << "Network not initializied properly \n";
         throw new std::logic_error("Structure missmatch");
    }
  
    size_t layerFrom = this -> inpDims[0] * this -> inpDims[1] * this -> inpDims[2];

    const size_t newDimsX = this -> inpDims[0] - (this -> kernelDims[0] - 1);
    const size_t newDimsY = this -> inpDims[1] - (this -> kernelDims[1] - 1);

    this -> outDims = { newDimsX, newDimsY, convolutions};

    if( this -> next -> type == ConvolutionalLayer ||  this -> next -> type == PoolingLayer ) {
        this -> next -> inpDims = this -> outDims;
    }
    else {
        this -> next -> neurons = vector<float>( newDimsX * newDimsY * this -> convolutions );
    }
    const size_t layerTo      = newDimsX * newDimsY * this -> convolutions;
    const size_t channels     = this -> inpDims[2]; 
    const size_t kernels      = channels * this -> convolutions;
    const size_t weightLength = kernels * this -> kernelDims[0] * this -> kernelDims[1];

    this -> neurons         = vector<float>( layerFrom );
    this -> intermed        = vector<float>( newDimsX * newDimsY * kernels );
    this -> summFeatureMaps = vector<float>( newDimsX * newDimsY * this ->convolutions );

    this -> errors          = vector<float>( layerFrom );
    this -> intermedErr     = vector<float>( layerFrom * this -> convolutions );
    this -> summErrors      = vector<float>( layerFrom );

    this -> weights         = vector<float>(weightLength);
    this -> dWcollect       = vector<float>(weightLength);
    this -> movAvg          = vector<float>(weightLength);
    this -> movExp          = vector<float>(weightLength);

    this -> bias            = vector<float>(convolutions); //ToDO: add option for tied bias
    this -> dwBiasCollect   = vector<float>(convolutions);
    this -> movAvgB          = vector<float>(convolutions);
    this -> movExpB          = vector<float>(convolutions);

    if( ! this->isInput ) {
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

void Conv2D::setInput(vector<float> & inp) {
    this -> neurons = inp;
};

void Conv2D::initAllRandom(const size_t &channels, const size_t &kernelX, const size_t &kernelY) {
    for(size_t i = 0; i < ( channels*this -> convolutions ); i++) {
        for(size_t j = 0; j < kernelY; j++) {
            for(size_t k = 0; k < kernelX; k++) {
                this->weights[i*kernelX*kernelY + j*kernelX + k] = 2*((float)rand()/(float)RAND_MAX) - 1;
            }
        }
        if(i < this-> convolutions) {
            this->bias[i] = 2*((float)rand()/(float)RAND_MAX) - 1;
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
            this->bias[i] = 2 * limit * ((float)rand()/(float)RAND_MAX) - limit;
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
            this->bias[i] = 2 * limit * ((float)rand()/(float)RAND_MAX) - limit;
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
            this->bias[i] = 2 * limit * ((float)rand()/(float)RAND_MAX) - limit;
        }
    }
}

void Conv2D::fwd() {
    this -> mathLib -> mtConv       ( this -> neurons , this -> weights, this -> intermed       , this -> inpDims , this -> outDims, this -> kernelDims, "conv_3d" );
    this -> mathLib -> mtConvAddBias( this -> intermed, this -> bias   , this -> summFeatureMaps, this -> outDims );
    this -> activate();
    this->next->fwd();
}

void Conv2D::bwd() {
    //calc dWs
    array<size_t, 3> outD = {  this -> kernelDims[0], this ->kernelDims[1], this -> outDims[2]};
    array<size_t, 2> kerD = {  this -> outDims[0], this ->outDims[1] };
    
    this->mathLib->mtConv( this-> neurons, this -> next -> errors, this->dWcollect, this -> inpDims, outD, kerD, "conv_3d_dw");
    //calc dWs Bias
    // since we only have this -> convolutions biases we can calculate this on cpu

    for( size_t i = 0; i < this -> convolutions; i++ ) {
        const size_t z = i * this -> outDims[0] * this -> outDims[1];
        for(size_t j = 0; j < this -> outDims[0] * this -> outDims[1]; j++) {
            this -> dwBiasCollect[i] += this -> next -> errors[z + j];
        }
    }

     if(!this->isInput) {
        //calc error for this layer to use in prev layer
        this -> mathLib -> mtConv    ( this -> next -> errors, this -> weights   , this->intermedErr, this -> outDims, this -> inpDims, this -> kernelDims, "conv_3d_bwd");
        this -> mathLib -> mtConvAdd ( this -> intermedErr   , this -> summErrors, this -> inpDims );
        this -> activateDW();
        this -> prev -> bwd();
    }
}

void Conv2D::learn(const float learnRate) {
    this -> optimize(learnRate);
    ++ this -> run;
    this->next->learn(learnRate);
}


void Conv2D::closs(vector<float> &target) {
    return;
}

void Conv2D::summary() {
            const string type = this -> type == ConvolutionalLayer ? "Convolutional Layer" : this -> type == DenseLayer ? "Dense Layer" : "Dense Output Layer";
            cout << type << ": \n";
            cout << "Neurons:\t\t" << this -> neurons.size()  << "\n" ;
            cout << "Weights:\t\t"  << this -> weights.size() << "\n" ;
            cout << "Biases:\t\t\t"   << this -> bias.size()  << "\n";
            cout << "Kernels: \t\t" << this -> inpDims[2] * this -> convolutions <<" per " << this -> kernelDims[0] << " x " << this -> kernelDims[1] <<  "\n";
            cout << "Total trainable params: " << ( this -> weights.size() + this -> bias.size() ) << "\n"; 
            cout << "Output feature maps:\t" << this -> convolutions  <<" per " << this -> outDims[0] << " x " << this -> outDims[1] <<  "\n";
            cout << "Input channels:\t\t" << this -> inpDims[2] <<" per " << this -> inpDims[0] << " x " << this -> inpDims[1] <<  "\n";
            cout << "_______________________________________________________________________________________\n\n";
            if ( this -> next != nullptr ) {
                this -> next -> summary();
            }
        }
#endif