#ifndef MODEL
#define MODEL

#include "./../layer/basic_layer.cpp"
#include "./../layer/dense_layer.cpp"
#include "stats.cpp"
#include <stdexcept>
#include <thread>
#include <type_traits>

template <class INP = float>
class Model {

    public:
        Model(float learnRate, vector<int> inpDim, bool autoInit, size_t batchSize, size_t epochs, size_t stepsPerEpoch);
        Model(vector<Layer*> layers, float learnRate, vector<int> inpDim, bool autoInit, size_t batchSize, size_t epochs, size_t stepsPerEpoch);
        Model(Model&  other);
        Model(Model&& other);
        Model();
        ~Model();

        Layer* first;
        Layer* last;
        Layer* current;

        float learnRate;
        int   inpNeurosn;
        bool  autoInit;

        size_t   batchssize;
        size_t   epochs;
        size_t   stepsPerEpoch;

        vector<int> inpDim;

        Stats stats;

        Model<INP>& operator=(Model other);
        
        void copyVals(Model& other);
        void fit(vector<vector<INP>> &inp, vector<vector<float>> &target);
        void compile();
        void add(Layer* next);
        void flattenInp(vector<vector<INP>> &inp, vector<vector<float>> &runs);
        void flattenOne(vector<vector<vector<float>>> &inp, vector<float> &runs);
        void flattenOne(vector<vector<float>> &inp, vector<float> &runs);
        void flattenOne(vector<float> &inp, vector<float> &runs);
        void printEpochStats(const size_t epoch) const;

        vector<float>& predict(vector<INP> &inp);
};

template <class INP >
Model<INP>::Model(float learnRate, vector<int> inpDim, bool autoInit, size_t batchSize , size_t epochs , size_t stepsPerEpoch ) {
    this->first      = nullptr;
    this->last       = nullptr;
    this->current    = nullptr;
    this->learnRate  = learnRate;
    this->inpNeurosn = 1;
    this->inpDim     = inpDim;
    this->autoInit   = autoInit;
    this->stepsPerEpoch = stepsPerEpoch;
    this->epochs     = epochs;
    this->batchssize = batchSize;
    this->stats      = Stats(stepsPerEpoch);

    for(size_t i = 0; i < inpDim.size(); i++) {
        this->inpNeurosn *= inpDim[i];
    }
}

template <class INP >
Model<INP>::Model(vector<Layer*> layers, float learnRate, vector<int> inpDim, bool autoInit, size_t batchSize, size_t epochs, size_t stepsPerEpoch) {
    this->first      = nullptr;
    this->last       = nullptr;
    this->current    = nullptr;
    this->learnRate  = learnRate;
    this->inpNeurosn = 1;
    this->inpDim     = inpDim;
    this->autoInit   = autoInit;
    this->stepsPerEpoch = stepsPerEpoch;
    this->batchssize = batchSize;
    this->stats      = Stats(stepsPerEpoch);

    for(size_t i = 0; i < inpDim.size(); i++) {
        this->inpNeurosn *= inpDim[i];
    }

    for(size_t i = 0; i < layers.size(); i++) {
        this -> add( layers[i] );
    }
}

template <class INP >
Model<INP>::Model() {
    this->first = nullptr;
    this->last  = nullptr;
    this->current = nullptr;
}

template <class INP >
Model<INP>::~Model() {
    if( this -> first == nullptr )
        return;
    this -> current = this -> first;
    while(this -> current -> next != nullptr) {
        this -> current = this -> current -> next;
        delete this -> current -> prev;
    }
    delete this -> current;
}

template <class INP >
Model<INP>& Model<INP>::operator=(Model<INP> other) {
    std::swap(first, other.first);
    std::swap(last, other.last);
    std::swap(current, other.current);
    copyContent(other);
    return *this;
}

template <class INP >
void Model<INP>::copyVals(Model<INP>& other) {
    this->learnRate = other.learnRate;
    this->inpDim    = other.inpDim;
    this->batchssize= other.batchssize;
    this->epochs    = other.epochs;
    this->stepsPerEpoch = other.stepsPerEpoch;
    this->autoInit  = other.autoInit;
}

template <class INP >
Model<INP>::Model(Model&  other) {
    if(other.first != nullptr)
        this->first = other.first->clone();
    if(other.last != nullptr)
        this->last  = other.last->clone();
    if(other.current != nullptr) 
        this->current = other.current->clone();
    this->copyVals(other);
}

template <class INP >
Model<INP>::Model(Model&& other) {
    this->first = other.first;
    other.first = nullptr;
    this->last  = other.last;
    other.last = nullptr;
    this->current = other.current;
    other.current = nullptr;
    this->copyVals(other);
}

template <class INP >
vector<float>& Model<INP>::predict(vector<INP> &inp) { 
    vector<float> run;
    run = vector<float>(inp.size());
    this->flattenOne(inp, run);

    this -> first -> setInput( run );
    this -> first -> fwd();
    return this -> last -> getOutput();
}

template <class INP >
void Model<INP>::fit(vector<vector<INP>> &inp, vector<vector<float>> &target) {
    cout << "Preparing Training data...\n";
    vector<vector<float>> runs;
    size_t currentEpoch = 0;
    //Todo: move statistic stuff into seperate thread .. thread statThread;
    runs = vector<vector<float>>(inp.size());
    this->flattenInp(inp, runs);
        
    cout << "Done, starting Training...\n";
    size_t currentEpochCount = 0;
    size_t currentBatchCount = 0;
    size_t radIdx            = 0;

    for(size_t epoch = 0; epoch < this->epochs; epoch++) {
        for(size_t step = 0; step < this -> stepsPerEpoch; step++) {

            radIdx = rand()%runs.size();

            this -> first -> setInput( runs[radIdx] );
           
            this->first->fwd();
            this->last->closs(target[radIdx]);

            ++currentBatchCount;
         
            this->stats.calcTotalLoss(target[radIdx], this -> last -> getOutput() );

            if(currentBatchCount >= this -> batchssize) {
                this -> first -> learn(this -> learnRate);
                currentBatchCount = 0;
            }
            
        }
        this->stats.eval();
        this->printEpochStats(epoch);
    }
}

template <class INP>
void Model<INP>::printEpochStats(const size_t epoch) const {
    cout << "Epoch " << epoch << " done.......... \n";
    cout << "Training Loss:     " << this->stats.currErr << "...... \n";
    cout << "Training Accuracy: " << this->stats.currHit << "...... \n \n";
}

template <class INP>
void Model<INP>::flattenInp(vector<vector<INP>> &inp, vector<vector<float>> &runs) {
    for( size_t i = 0; i < inp.size(); i++) {
        this -> flattenOne( inp[i], runs[i]);
    }
}

template <class INP>
void Model<INP>::flattenOne(vector<vector<vector<float>>> &inp, vector<float> &runs) {
    runs = vector<float>( inpDim[0] * inpDim[1] * inpDim[2] );
    for(size_t i = 0; i < inpDim[2]; i++ ) {
        for(size_t j = 0; j < inpDim[1]; j++ ) {
            for(size_t k = 0; k < inpDim[0]; k++ ) {
                runs[i * inpDim[1] * inpDim[0] + j*inpDim[0] + k] = inp[i][j][k];
            }        
        } 
    } 
}

template <class INP>
void Model<INP>::flattenOne(vector<vector<float>> &inp, vector<float> &runs) {
    runs = vector<float>( inpDim[0] * inpDim[1] );
    for(size_t j = 0; j < inpDim[1]; j++ ) {
        for(size_t k = 0; k < inpDim[0]; k++ ) {
            runs[j*inpDim[0] + k] = inp[j][k];
        }        
    } 
}

template <class INP>
void Model<INP>::flattenOne(vector<float> &inp, vector<float> &runs) {
    runs = inp;
}

template<class INP>
void  Model<INP>::add(Layer* const next) {
    if(this->first == nullptr) {
        this->first = next;
        this->current = next;
    } else {
        this->current->next = next;
        this->current->next->prev = this->current;
        this->current = next;
        this->last    = next;
    }
}

template<class INP>
void Model<INP>::compile() {
    if(this -> first != nullptr) {
        this -> first -> setupLayer();
        this -> stats.evalLoss(this -> last -> loss);
        this -> first -> summary();
    } else {
        cout << "No layers provided yet \n";
    }
}
#endif