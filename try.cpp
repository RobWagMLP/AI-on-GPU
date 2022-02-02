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
#include <cl_math_lib.cpp>

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
    const int heightB = 5;
    const int widthB = 1;
    cout << "Multiplying 128x128 Matrices..\n";
    try{
        ClMathLib* lib = new ClMathLib();
        std::vector<float> h_a(widthA*heightA), h_b(widthB * heightB), h_d(widthA*heightB), h_c(widthA*heightB);
    
        for (size_t i = 0; i < widthA*heightA; i++) {
            h_a[i] = (float)i;
        }
        for (size_t i = 0; i < widthB * heightB; i++) {
            h_b[i] = (i*i)%10;
        }
        for (size_t i = 0; i < widthB * heightB; i++) {
            h_d[i] = 1;
        }
        h_c = lib->vcErr(h_a, h_b, "relu_cat_crent" );
        printMat(h_a, widthA, heightA);
        cout << "\n";
        printMat(h_b, heightB, widthB);
        cout << "\n";
        //printMat(h_b, widthB, heightB);
        printMat(h_c, widthA, widthB);
        delete lib;
    } catch(cl::Error er) {
        cout << er.err() << "\n";
    }
}

