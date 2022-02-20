#ifndef MODEL
#define MODEL

#include "./../layer/basic_layer.cpp"
#include "./../layer/dense_layer.cpp"
#include "stats.cpp"
#include <stdexcept>
#include <thread>

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
        void fit(const vector<vector<INP>> &inp, vector<vector<float>> &target);
        void compile();
        void add(Layer* next);
        void flattenInp(const vector<vector<INP>> &inp, vector<vector<float>> &runs);
        void flattenOne(const vector<INP> &inp, vector<float> &runs);
        void printEpochStats(const size_t epoch) const;

        vector<float> predict(const vector<INP> &inp);
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
vector<float> Model<INP>::predict(const vector<INP> &inp) { 
    vector<float> run;
    if(this->inpDim.size() == 1) {
        run = inp;
    } else {
        run = vector<float>(inp.size());
        this->flattenOne(inp, run);
    }

    if(run.size() != this->first->neurons.size()) {
        cout << "Cant't start Predicting. Input doesnt match Network structure \n";
        throw new std::logic_error("Structure missmatch");
    }

    this -> first -> neurons = inp;
    this -> first -> fwd();
    return this -> last -> neurons;
}

template <class INP >
void Model<INP>::fit(const vector<vector<INP>> &inp, vector<vector<float>> &target) {
    cout << "Preparing Training data...\n";
    vector<vector<float>> runs;
    size_t currentEpoch = 0;
    //Todo: move statistic stuff into seperate thread .. thread statThread;
    if(this->inpDim.size() == 1) {
        runs = inp;
    } else {
        runs = vector<vector<float>>(inp.size());
        this->flattenInp(inp, runs);
    }
     if(runs[0].size() != this->first->neurons.size()) {
        cout << "Cant't start Training. Input doesnt match Network structure \n";
        throw new std::logic_error("Structure missmatch");
    }

    if(target[0].size() != this->last->neurons.size()) {
        cout << "Cant't start Training. Output doesnt match Network structure \n";
        throw new std::logic_error("Structure missmatch");
    }
    
    cout << "Done, starting Training...\n";
    size_t currentEpochCount = 0;
    size_t currentBatchCount = 0;
    size_t radIdx            = 0;

    for(size_t epoch = 0; epoch < this->epochs; epoch++) {
        for(size_t step = 0; step < this -> stepsPerEpoch; step++) {

            radIdx = rand()%runs.size();

            this->first->neurons = runs[radIdx];
           
            this->first->fwd();
            this->last->closs(target[radIdx]);

            ++currentBatchCount;

            this->stats.calcTotalLoss(target[radIdx], this->last->neurons);

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
void Model<INP>::flattenInp(const vector<vector<INP>> &inp, vector<vector<float>> &runs) {

}

template <class INP>
void Model<INP>::flattenOne(const vector<INP> &inp, vector<float> &runs) {

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
    } else {
        cout << "No layers provided yet \n";
    }
}
#endif