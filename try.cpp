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
#include <dense_out.cpp>

using namespace std;

 const string loadprogram(string path) {
    std::ifstream in(path);
    std::stringstream buffer;
    std::string line;
    while (std::getline(in, line)) {
        buffer << line;
    }
    
    std::string contents(buffer.str());
    in.close();
    return contents;
 }

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
    const int heightB = 1;
    const int widthB = 5;
    cout << "Multiplying 128x128 Matrices..\n";
    try{
        DenseOut out(5, CATEGORICAL_CROSS_ENTROPY, SOFTMAX, new DenseOut(10, MEAN_SQUARED, TANH, nullptr));
        DenseOut out2 = (out);
        std::shared_ptr<ClMathLib> lib = ClMathLib::instanceML();
        
        std::vector<float> h_a(widthA*heightA), h_b(widthB * heightB), h_d(widthA*heightB), h_c(heightA*widthB);

        for (size_t i = 0; i < widthA*heightA; i++) {
            h_a[i] = (float)(i%3);
        }
        for (size_t i = 0; i < widthB * heightB; i++) {
            h_b[i] = (i*i)%10;
        }
        for (size_t i = 0; i < widthB * heightB; i++) {
            h_d[i] = 1;
        }
        //lib->mtPrd(h_a, h_b, h_c, heightA, widthB );
        out2.neurons = h_a;
        out2.closs(h_b);
        h_c = out2.errors;
        printMat(out2.neurons, widthA, heightA);
        cout << "\n";
        printMat(h_b, heightB, widthB);
        cout << "\n";
        //printMat(h_b, widthB, heightB);
        printMat(h_c, widthA, heightA);
        cout <<out2.tot_errs[0] << "\n";
    } catch(cl::Error er) {
        cout << er.err() << "\n";
    }
}

