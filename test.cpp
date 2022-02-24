#define __CL_ENABLE_EXCEPTIONS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include "layer/dense_layer.cpp"
#include "layer/dense_out.cpp"
#include "layer/conv2d_layer.cpp"
#include "model/model.cpp"

using namespace std;


vector<vector<float>> targ = { { 1 }, { 0 }, { 1 }, { 0 }, { 1 }, { 0 }, { 0 }, { 1 }, { 0 } , { 1 }};

vector<vector<vector<float>>> inp = {
    {//T
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    {//F
        { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    {//T
        { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0},
        { 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
        { 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    },
     {//F
        { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0},
        { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1},
        { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1},
        { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
     {//T
        { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        { 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0},
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        { 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    },
     {//F
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
     {//F
        { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        { 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0},
        { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0},
        { 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        { 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    },
     {//T
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0},
        { 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
        { 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1},
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0},
        { 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0},
    },
    {//F
        { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0},
        { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0},
        { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        { 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0},
        { 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        { 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    },
     {//T
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        { 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},
        { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    },
};

vector<float> multMatCPU(vector<float> h_a, vector<float> h_b, int aHeight, int bWidth) {
    auto t1 = chrono::high_resolution_clock::now();

    cout << "starting on cpu.. \n";
    const int aWidth = h_a.size()/aHeight;
    vector<float> c(aHeight*bWidth); 
    for(int i = 0; i < aHeight; i++) {
        for(int j = 0; j < bWidth; j++) {
            float tmp = 0;
            for(int k = 0; k < aWidth; k++) {
                 tmp += h_a[i*aWidth+k] * h_b[k*bWidth+j];
            }
            c[i*bWidth + j] = tmp;
        }
    }
    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
    cout << "Done in " << ms_int.count() <<" milliseconds \n";
    return c;
}

void printMat(vector<float> mat, int widthA, int heightA) {
 for(int i = 0; i < heightA*widthA; i++) {
        cout << mat[i] << ",";
        if((i + 1)%widthA == 0) {
            cout << "\n";
        }
    }
}

void printMultMat(vector<float> mat, int width, int height, int amount) {
    for(int i = 0; i < amount; i++) {
        for(int j = i*width*height; j < (i+1)*width*height; j++) {
            cout << mat[j] << ",";
            if((j + 1)%width == 0) {
                cout << "\n";
            }
        }
        cout <<"\n";
    }
}

 int main( int argc, char* argv[] )
{
    try{
        Model<vector<float>> model( 0.05, { 12, 12 }, true, 4, 100, 50);
        model.add(new Conv2D  ( {12, 12, 1 }, { 3, 3 }, 1, RELU, GAUSIAN         ) );
        model.add(new Dense   (     RELU   , GAUSIAN ) );
        model.add(new Dense   ( 28, SIGMOID, GAUSIAN ) );
        model.add(new DenseOut( 1, BINARY_CATEGORICAL_CROSS_ENTROPY , nullptr   ) );
        model.compile();
        
        
        model.fit(inp, targ);
/*
        vector<float> &res = model.predict(inp[0]);
        printMat(res, res.size(), 1);
        res = model.predict(inp[1]);
        printMat(res, res.size(), 1);
        res = model.predict(inp[2]);
        printMat(res, res.size(), 1);
        res = model.predict(inp[3]);
        printMat(res, res.size(), 1);
            /*
        
        array<size_t, 3> inDims =  { 4, 4, 2 };
        array<size_t, 3> outDims = { 6, 6, 2 };
        array<size_t, 2> kerDims = { 3, 3};

        vector<float> mat(inDims[0] * inDims[1] * inDims[2]);
        vector<float> ker(kerDims[0] * kerDims[1] * outDims[2] * inDims[2]);
        vector<float> outp(outDims[0]*outDims[1] * inDims[2] * outDims[2]);

        shared_ptr<ClMathLib> lib = ClMathLib::instanceML();

        for(int i = 0; i < mat.size(); i++) {
            mat[i] = i%4 + 1;
        }
        for(int i = 0; i < ker.size(); i++) {
            ker[i] = i%3;
        }
        lib -> mtConv(mat, ker, outp, inDims, outDims, kerDims, "conv_3d_bwd");

        printMultMat(mat, inDims[0], inDims[1], inDims[2]);
        printMultMat(ker, kerDims[0], kerDims[1], outDims[2]* inDims[2]);
        printMultMat(outp, outDims[0], outDims[1], inDims[2]* outDims[2]);
        vector<float> addres(outDims[0] * outDims[1] * outDims[2]);
        vector<float> bias = {1, 1};
        lib -> mtConvAdd(outp, addres, outDims);
        printMultMat(addres, outDims[0], outDims[1], outDims[2]);
        lib -> mtConvAddBias(outp, bias, addres, outDims);*/
        

    } catch(cl::Error er) {
        cout << er.err() << "\n";
    }
}

