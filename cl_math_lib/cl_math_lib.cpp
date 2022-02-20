#ifndef CLMATHLIB
#define CLMATHLIB

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

/**
 * @brief core concept is:
 * create a cl-context
 * create cl queue from context
 * create programs
 * 
 * in methods:
 * create buffers for data to pass to gpu
 * create kernel functor, that correspons to defined interface in the kernel method
 * call and copy result into output
 * 
 */

class ClMathLib {
    private: 
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program programVecAdd;
        cl::Program programMatMult;
        cl::Program programVecAct;
        cl::Program programMultDyad;
        cl::Program programMultBias;
        cl::Program programVecDwMt;
        cl::Program programVecErr;
        cl::Program programVcAddFct;
        cl::Program programConv3D;
        ClMathLib();

    public: 
        string loadProgram(const string &program);
        void   vcAdd    (vector<float> &h_a,  vector<float> &h_b,    vector<float> &h_c, string method);
        void   vcAddFct (vector<float> &h_a,  vector<float> &h_b,    vector<float> &h_c, const float lrnRt);
        void   mtPrd    (vector<float> &h_a,  vector<float> &h_b,    vector<float> &h_c, int aHeight, int bWidth);
        void   mtConv   (vector<float> &h_in, vector<float> &h_kern, vector<float> &h_out, array<size_t, 3> &inpDims, array<size_t, 3> &outpDims, array<size_t, 2> &kerDims);
        void   mtPrdBias(vector<float> &h_a,  vector<float> &h_b,    vector<float> &h_d, vector<float> &h_c, int aHeight, int bWidth);
        void   vcAct    (vector<float> &h_a,  vector<float> &h_c,    string method);
        void   vcDwMt   (vector<float> &h_a,  vector<float> &h_b,    vector<float> &h_c, string method);
        void   vcErr    (vector<float> &h_a,  vector<float> &h_b,    vector<float> &h_c, string method);
        void   dVcAct   (vector<float> &h_a,  vector<float> &h_c,    string method);
        void   mtDyad   (vector<float> &h_a,  vector<float> &h_c,    vector<float> &h_b);
        //Singleton, which is okay, since we have no parameters that need to be accessed
        ClMathLib(ClMathLib const&) = delete;
        
        ClMathLib& operator=(ClMathLib const&) = delete;
        
        static std::shared_ptr<ClMathLib> instanceML() {
            static std::shared_ptr<ClMathLib> s{new ClMathLib};
            return s;
        }
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
programVecDwMt(context,
               loadProgram("./cl_math_lib/programs/vec_dw_mt.cl"), 
               true),
programVecErr(context,
               loadProgram("./cl_math_lib/programs/vec_err.cl"), 
               true),
programVcAddFct(context,
               loadProgram("./cl_math_lib/programs/vec_add_fct.cl"), 
               true),
programConv3D(context,
               loadProgram("./cl_math_lib/programs/conv_3d.cl"), 
               true)
 {}


 string ClMathLib::loadProgram(const string &program) {
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

 void ClMathLib::vcAdd(vector<float> &h_a, vector<float> &h_b, vector<float> &h_c, string method = "vadd") {
    const size_t LENGTH = h_a.size();
    int errcode_ret = 0;
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
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

  void ClMathLib::vcAddFct(vector<float> &h_a, vector<float> &h_b, vector<float> &h_c, const float lrnRt) {
    const size_t LENGTH = h_a.size();
    int errcode_ret = 0;
    cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
    cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*LENGTH);
    cl::Buffer d_d(context, CL_MEM_WRITE_ONLY,  sizeof(float)*LENGTH);
    try {
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, float, int> vadd(programVcAddFct, "vadd", &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(LENGTH)),
                    d_a, d_b, d_c, d_d, lrnRt, LENGTH);

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        cl::copy(queue, d_d, h_b.begin(), h_b.end());
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

 void ClMathLib::vcAct(vector<float> &h_a, vector<float> &h_c, string method) {
    const size_t LENGTH = h_a.size();
    int errcode_ret = 0;
    cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*LENGTH);
    try {
        cl::KernelFunctor<cl::Buffer, cl::Buffer, int> vadd(programVecAct, method, &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(LENGTH)),
                    d_a, d_c, LENGTH);

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

void ClMathLib::vcDwMt(vector<float> &h_a, vector<float> &h_b, vector<float> &h_c, string method) {
    const size_t LENGTH = h_a.size();
    int errcode_ret = 0;
    cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
    cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
    cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*LENGTH);
    try {
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(programVecDwMt, method, &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(LENGTH)),
                    d_a, d_b, d_c, LENGTH);

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

  void ClMathLib::vcErr(vector<float> &h_a, vector<float> &h_b, vector<float> &h_c, string method) {
    const size_t LENGTH = h_a.size();
    int errcode_ret = 0;
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
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

 void ClMathLib::mtPrd(vector<float> &h_a, vector<float> &h_b, vector<float> &h_c, int aHeight, int bWidth) {
    const int cLength = aHeight*bWidth;
    const int aWidth = h_a.size()/aHeight;
    int errcode_ret = 0;
    auto t0 = chrono::high_resolution_clock::now();
    try {
        auto t1 = chrono::high_resolution_clock::now();
        auto ms_init = chrono::duration_cast<chrono::milliseconds>(t1 - t0);
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
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }
  void ClMathLib::mtConv(vector<float> &h_in, vector<float> &h_kern, vector<float> &h_out, array<size_t, 3> &inpDims, array<size_t, 3> &outpDims, array<size_t, 2> &kerDims) {
    int errcode_ret = 0;
    size_t length = h_out.size();
    try {
        cl::Buffer d_a(context, h_in.begin(), h_in.end(), true);
        cl::Buffer d_b(context, h_kern.begin(), h_kern.end(), true);
        cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*length);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int,int, int, int > vadd(programConv3D, "mat_mul", &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange( outpDims[0], outpDims[1], outpDims[2] * inpDims[2] ) ), // for n*m outputchannel we have channels * convolutions intermediate featuremaps, that get add up together later.
                    d_a, d_b, d_c, outpDims[0], outpDims[1], inpDims[0], inpDims[1], inpDims[2], kerDims[0], kerDims[1] );

        cl::copy(queue, d_c, h_out.begin(), h_out.end());
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

void ClMathLib::mtPrdBias(vector<float> &h_a, vector<float> &h_b, vector<float> &h_d, vector<float> &h_c, int aHeight, int bWidth) {
    const int cLength = aHeight*bWidth;
    const int aWidth = h_a.size()/aHeight;
    int errcode_ret = 0;
    try {
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
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }


void ClMathLib::mtDyad(vector<float> &h_a, vector<float> &h_b,  vector<float> &h_c) {
    const int cLength = h_a.size() * h_b.size();
    int errcode_ret = 0;
    try {
        cl::Buffer d_a(context, h_a.begin(), h_a.end(), true);
        cl::Buffer d_b(context, h_b.begin(), h_b.end(), true);
        cl::Buffer d_d(context, h_c.begin(), h_c.end(), true);
        cl::Buffer d_c(context, CL_MEM_WRITE_ONLY,  sizeof(float)*cLength);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, size_t, size_t> vadd(programMultDyad, "mat_dyad", &errcode_ret);
        vadd(cl::EnqueueArgs(
                    queue, 
                    cl::NDRange(h_a.size(), h_b.size())),
                    d_a, d_b, d_d, d_c, h_a.size(), h_b.size());

        cl::copy(queue, d_c, h_c.begin(), h_c.end());
    } catch(cl::Error er) {
        cout << er.err() << ", " << errcode_ret;
        throw(er);
    }
 }

 #endif