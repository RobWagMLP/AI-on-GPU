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
#include "layer/pooling_layer.cpp"
#include "model/model.cpp"
#include <CImg.h>

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

vector<vector<float>> testC = {
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        { 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        { 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
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
        cimg_library::CImg<unsigned char> src("./index.jpeg");
        int width = src.width();
        int height = src.height();
        cout << width << "x" << height << endl;
        cout << "spectrum: " << src.spectrum() <<"\n";
        return 0;
        /*
        Model<vector<float>> model( 0.01, { 12, 12 }, true, 32, 50, 50);

        model.add(shared_ptr<Layer>(new Conv2D  ( {12, 12, 1 }, { 3, 3 }, 3, RELU, GAUSIAN, ADAM  ) ) );
        model.add(shared_ptr<Layer>(new Pooling (               { 2, 2 }, MAX                     ) ) );
        model.add(shared_ptr<Layer>(new Dense   (                            RELU, GAUSIAN, ADAM  ) ) );
        model.add(shared_ptr<Layer>(new Dense   ( 28,                     SIGMOID, GAUSIAN, ADAM  ) ) );
        model.add(shared_ptr<Layer>(new DenseOut( 1 , BINARY_CATEGORICAL_CROSS_ENTROPY            ) ) );
        model.compile();
              
        model.fit(inp, targ);

        vector<float> &res = model.predict(testC);
        printMat(res, res.size(), 1);
        res = model.predict(inp[1]);
        printMat(res, res.size(), 1);
        res = model.predict(inp[2]);
        printMat(res, res.size(), 1);
        res = model.predict(inp[3]);
        printMat(res, res.size(), 1);
            /*
        
        array<size_t, 3> inDims =  { 10, 10, 2 };
        array<size_t, 3> outDims = { 5, 5, 2 };
        array<size_t, 2> kerDims = { 2, 2};

        vector<float> mat(inDims[0] * inDims[1] * inDims[2]);
        vector<float> indi(inDims[0] * inDims[1] * inDims[2]);
        vector<float> outp(outDims[0]*outDims[1] * outDims[2]);

        shared_ptr<ClMathLib> lib = ClMathLib::instanceML();

        for(int i = 0; i < mat.size(); i++) {
            mat[i] = i%9 + 1;
        }

        lib -> mtPoolMinMax(mat, outp, indi, inDims, outDims, kerDims, 1);

        printMultMat(mat, inDims[0], inDims[1], inDims[2]);
        printMultMat(outp, outDims[0], outDims[1], outDims[2]);
        printMultMat(indi, outDims[0], outDims[1], outDims[2]);*/
    } catch(cl::Error &er) {
        cout << er.err() << "\n";
        throw er;
    }
}

