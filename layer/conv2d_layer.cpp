#ifndef CONV2D
#define CONV2D

#include "basic_layer.cpp"

class Conv2D: public Layer {
    public:
        Conv2D(){};
        Conv2D(vector<size_t> inpDims, Activation iAct, WeightInit iWeightInit);
        Conv2D(Activation iAct, WeightInit iWeightInit);
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

#endif