#include <basic_layer.cpp>

class DenseOut: public Layer {
    public:
        DenseOut(Loss iLoss, Activation iAct, Layer *iPrev);
        void closs();

};

DenseOut::DenseOut(Loss iLoss, Activation iAct, Layer *iPrev) {
    loss        = iLoss;
    activation  = iAct;
    prev        = iPrev;
    weights     = nullptr;
    dW          = nullptr;     
}

DenseOut::~DenseOut() {
    delete[] neurons;
    delete[] prev;
}

void DenseOut::closs() {
    switch(loss) {
        case MEAN_SQUARED                       : return;
        case CATEGORICAL_CROSS_ENTROPY          : return;
        case SPARE_CATEGORICAL_CROSS_ENTROPY    : return;
        case BINARY_CATEGORICAL_CROSS_ENTROPY   : return;
        case MULTI_CLASS_CROSS_ENTROPY          : return;
        default                                 : return;
    };
}