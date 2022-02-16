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
#include "model/model.cpp"

using namespace std;

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

 int main( int argc, char* argv[] )
{
    const int widthA = 5;
    const int heightA = 1;
    const int heightB = 10;
    const int widthB = 1;
    try{
        Model<float> model( 0.05, {1}, true, 4, 100, 50);
        model.add(new Dense   ( 2, RELU           , nullptr, nullptr, GAUSIAN      ) );
        model.add(new Dense   ( 5, SIGMOID        , nullptr, nullptr , GAUSIAN ) );
        model.add(new DenseOut( 1, BINARY_CATEGORICAL_CROSS_ENTROPY , nullptr     ) );
        model.compile();
        vector<vector<float>> in = { {0, 0 }, {0, 1}, {1, 0}, {1, 1.} };
        vector<vector<float>> out = {{0}, {1.}, {1.}, {0.} };
       // vector<vector<float>> in = { {1.}, {2.}, {3.}, {4.}, {5.} };
       // vector<vector<float>> out = {{1.,0.,0.,0.,0.}, {0.,1.,0.,0.,0.}, {0.,0.,1.,0.,0.}, {0.,0.,0.,1.,0.}, {0.,0.,0.,0.,1.}};
        model.fit(in, out);

        vector<float> res = model.predict(in[0]);
        printMat(res, res.size(), 1);
        res = model.predict(in[1]);
        printMat(res, res.size(), 1);
        res = model.predict(in[2]);
        printMat(res, res.size(), 1);
        res = model.predict(in[3]);
        printMat(res, res.size(), 1);
    } catch(cl::Error er) {
        cout << er.err() << "\n";
    }
}

