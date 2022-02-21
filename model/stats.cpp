#ifndef STATS
#define STATS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "./../layer/enums.cpp"
#include <functional>
#include "./../config/constants.h"

using namespace std;

class Stats {
    public:
        std::vector<float> errorHistory;
        std::vector<float> hitCountHistory;
        float         currErrSum;
        float         currHitSum;
        float         currErr   ;
        float         currHit   ;
        size_t stepsPerRun;
        Stats();
        Stats(size_t stepsPerRun):
        errorHistory(0),
        hitCountHistory(0) {
            this->stepsPerRun = stepsPerRun;
            currErrSum = 0;
            currHitSum = 0;
        }

        void eval();

        void totalLossMeanSq    (const vector<float> &target, const vector<float> &acts  );
        void totalLossbinCross  (const vector<float> &target, const vector<float> &acts  );
        void totalLossCrEnt     (const vector<float> &target, const vector<float> &acts  );
        void totalLossSprsCrEnt (const vector<float> &target, const vector<float> &acts  );
        void evalLoss(Loss loss);

        std::function<void(const vector<float>&, const vector<float>&)> calcTotalLoss;
};

Stats::Stats() {
    this -> stepsPerRun = 0;
}

void Stats::evalLoss(Loss loss) {
    switch(loss) {
        case(MEAN_SQUARED):                    this->calcTotalLoss = [this](const vector<float> &target, const vector<float> &acts  ) {this->totalLossMeanSq(target, acts); };
                                               break;
        case(BINARY_CATEGORICAL_CROSS_ENTROPY):
        case(MULTI_CLASS_CROSS_ENTROPY):       this->calcTotalLoss = [this](const vector<float> &target, const vector<float> &acts  ) {this->totalLossbinCross(target, acts); };
                                               break;
        case(CATEGORICAL_CROSS_ENTROPY):       this->calcTotalLoss = [this](const vector<float> &target, const vector<float> &acts  ) {this->totalLossCrEnt(target, acts); };
                                               break;
        case(SPARE_CATEGORICAL_CROSS_ENTROPY): this->calcTotalLoss = [this](const vector<float> &target, const vector<float> &acts  ) {this->totalLossSprsCrEnt(target, acts); };
                                               break;   
    }
}

void Stats::totalLossMeanSq(const vector<float> &target, const vector<float> &acts  ) {
    float error = 0.;
    bool hit = true;
    for(size_t i = 0; i < acts.size(); i++) {
        error += pow( (acts[i] - target[i] ), 2);
        if(abs(target[i] - acts[i]) > HIT_COUNT_LIMIT) {
            hit = false;
        }
    }
    this->currHitSum += hit ? 1. : 0.;
   
    this->currErrSum += (error/(float)acts.size() );
}

void Stats::totalLossbinCross(const vector<float> &target, const vector<float> &acts ) {
    float error = 0;
    bool hit = true;
    for(size_t i = 0; i < acts.size(); i++) {
        error -= target[i] * log( NON_ZERO_CONSTANT_STAT + acts[i] ) + ( 1 - target[i])*log( NON_ZERO_CONSTANT_STAT + 1 - acts[i]);
        if(abs(target[i] - acts[i]) > HIT_COUNT_LIMIT) {
            hit = false;
        }
    }
    this->currHitSum += hit ? 1. : 0.;
    this->currErrSum += (error/(float)acts.size() );
}
void Stats::totalLossCrEnt(const vector<float> &target, const vector<float> &acts ) {
    float error = 0;
    bool hit = true;
    for(size_t i = 0; i < acts.size(); i++) {
        error -= target[i] * log( acts[i] + NON_ZERO_CONSTANT_STAT);
        if(abs(target[i] - acts[i]) > HIT_COUNT_LIMIT) {
            hit = false;
        }
    }
    this->currHitSum += hit ? 1. : 0.;
    this->currErrSum += (error/(float)acts.size() );
}

void Stats::totalLossSprsCrEnt(const vector<float> &target,const vector<float> &acts ) {
    this->currErrSum += -log( acts[target[0]] );
    if(abs(target[0] - acts[0]) > HIT_COUNT_LIMIT) {
            this->currHitSum += 1;
        }
}

void Stats::eval() {
    this->currErr = this->currErrSum / this->stepsPerRun;
    this->currHit = this->currHitSum / this->stepsPerRun;
    this->errorHistory.push_back(this->currErr);
    this->hitCountHistory.push_back(this->currHit);
    this->currErrSum = 0;
    this->currHitSum = 0;
}

#endif