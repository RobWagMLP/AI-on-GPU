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

using namespace std;


class ClMathLib {
    private: 
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program programVecAdd;
        cl::Program programMatMult;
        cl::Program programVecAct;
        cl::Program programMultDyad;
        cl::Program programMultBias;
        cl::Program programVecErr;

    public: 
        ClMathLib();
        string loadProgram(string program);
        vector<float>  vcAdd(vector<float> &h_a, vector<float> &h_b, string method);
        vector<float>  mtPrd(vector<float> &h_a, vector<float> &h_b, int aHeight, int bWidth);
        vector<float>  mtPrdBias(vector<float> &h_a, vector<float> &h_b, vector<float> &h_d, int aHeight, int bWidth);
        vector<float>  vcAct(vector<float> &h_a, string method);
        vector<float>  vcErr(vector<float> &h_a, vector<float> &h_b, string method);
        vector<float> dVcAct(vector<float> &h_a, string method);
        vector<float> mtDyad(vector<float> &h_a, vector<float> &h_b);
};

ClMathLib::ClMathLib()
:context(CL_DEVICE_TYPE_GPU), 
 queue(context),
 programVecAdd(context,
               loadProgram("./cl_math_lib/programs/vec_add.cl"), 
               true),
 programMatMult(context,
               loadProgram("./cl_math_lib/programs/mat_prod.cl"), 
               true),
 programVecAct(context,
               loadProgram("./cl_math_lib/programs/vec_act.cl"), 
               true),
programMultDyad(context,
               loadProgram("./cl_math_lib/programs/mat_dyad.cl"), 
               true),
programMultBias(context,
               loadProgram("./cl_math_lib/programs/mat_prod_bias.cl"), 
               true),
programVecErr(context,
               loadProgram("./cl_math_lib/programs/vec_err.cl"), 
               true)
 {}

 string ClMathLib::loadProgram(string program) {
    std::ifstream in(program);
    std::stringstream buffer;
    std::string line;
    while (std::getline(in, line)) {
        buffer << line;
    }
    
    std::string contents(buffer.str());
    in.close();
    return contents;
 }

 vector<float> ClMathLib::vcAdd(vector<float> &h_a, vector<float> &h_b, string method = "vadd") {
    const size_t LENGTH = h_a.size();
    int errcode_ret = 0;
    std::vector<float> h_c(LENGTH);  
    cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
    cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*LENGTH);
    try {
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(programVecAdd, method, &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(LENGTH)),
                    d_a, d_b, d_c, LENGTH);

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        return h_c;
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

 vector<float> ClMathLib::vcAct(vector<float> &h_a, string method) {
    const size_t LENGTH = h_a.size();
    int errcode_ret = 0;
    std::vector<float> h_c(LENGTH);  
    cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*LENGTH);
    try {
        cl::KernelFunctor<cl::Buffer, cl::Buffer, int> vadd(programVecAct, method, &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(LENGTH)),
                    d_a, d_c, LENGTH);

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        return h_c;
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

  vector<float> ClMathLib::vcErr(vector<float> &h_a, vector<float> &h_b, string method) {
    const size_t LENGTH = h_a.size();
    int errcode_ret = 0;
    std::vector<float> h_c(LENGTH);  
    cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
    cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*LENGTH);
    try {
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(programVecErr, method, &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(LENGTH)),
                    d_a, d_b, d_c, LENGTH);

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        return h_c;
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

 vector<float> ClMathLib::mtPrd(vector<float> &h_a, vector<float> &h_b, int aHeight, int bWidth) {
    const int cLength = aHeight*bWidth;
    const int aWidth = h_a.size()/aHeight;
    int errcode_ret = 0;
    auto t0 = chrono::high_resolution_clock::now();
    cout << "init gpu.. \n";
    try {
        std::vector<float> h_c(cLength);
        auto t1 = chrono::high_resolution_clock::now();
        auto ms_init = chrono::duration_cast<chrono::milliseconds>(t1 - t0);
        cout << "Initialized in " << ms_init.count() <<" milliseconds \n";
        cout << "starting on gpu.. \n";
        cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
        cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
        cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*cLength);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int> vadd(programMatMult, "mat_mul", &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(aHeight, bWidth)),
                    d_a, d_b, d_c, aWidth, bWidth);

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        auto t2 = chrono::high_resolution_clock::now();
        auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
        cout << "Done in " << ms_int.count() <<" milliseconds \n";
        return h_c;
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

vector<float>  ClMathLib::mtPrdBias(vector<float> &h_a, vector<float> &h_b, vector<float> &h_d, int aHeight, int bWidth) {
    const int cLength = aHeight*bWidth;
    const int aWidth = h_a.size()/aHeight;
    int errcode_ret = 0;
    cout << "here";
    try {
        std::vector<float> h_c(cLength);

        cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
        cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
        cl::Buffer d_d(context, h_d.begin(), h_d.end(), true);
        cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*cLength);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> vadd(programMultBias, "mat_mul_bias", &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(aHeight, bWidth)),
                    d_a, d_b, d_d,  d_c, aWidth, bWidth);

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        return h_c;
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }


vector<float> ClMathLib::mtDyad(vector<float> &h_a, vector<float> &h_b) {
    const int cLength = h_a.size() * h_b.size();
    int errcode_ret = 0;
    auto t0 = chrono::high_resolution_clock::now();
    try {
        std::vector<float> h_c(cLength);
        cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
        cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
        cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*cLength);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, size_t, size_t> vadd(programMultDyad, "mat_dyad", &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(h_a.size(), h_a.size())),
                    d_a, d_b, d_c, h_a.size(), h_a.size());

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        return h_c;
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }