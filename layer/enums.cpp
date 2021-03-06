#ifndef ENUMS
#define ENUMS
enum Loss {
    MEAN_SQUARED,
    CATEGORICAL_CROSS_ENTROPY,
    BINARY_CATEGORICAL_CROSS_ENTROPY,
    SPARE_CATEGORICAL_CROSS_ENTROPY,
    MULTI_CLASS_CROSS_ENTROPY
};

enum Activation {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX
};

enum WeightInit {
    ALLRANDOM,
    XAVIERUNIFORM,
    XAVIERNORMAL,
    GAUSIAN
};

enum LayerType {
    OutputLayer,
    DenseLayer,
    ConvolutionalLayer,
    PoolingLayer
};

enum Optimization {
    SGD,
    ADAM
};

enum PoolingType {
    MAX,
    MIN,
    AVG
};

#endif