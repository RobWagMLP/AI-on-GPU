#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
 
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelMult =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] * b[id] ;                                  \n" \
"}                                                               \n" \
                                                                "\n" ;

const char *kernelAdd =                                          "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id] ;                                  \n" \
"}                                                               \n" \
                                                                "\n" ;

const char *kernelInit =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecInit( __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n) {                                                \n" \
"        a[id] = 1;                                              \n" \
"        b[id] = 1;                                           \n" \
"    }                                                           \n" \
"}                                                               \n" \
                                                                "\n" ;
 
int main( int argc, char* argv[] )
{

    std::ifstream in("shader.cpp");
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string contents(buffer.str());
    // Length of vectors
    unsigned int n = 16;
    unsigned int stride = 4;

    
    // Host input vectors
    double *h_a;
    double *h_b;
    for (int i = 0; i < n; i++) {
        h_a[i] = 1;
        h_b[i] = 2;
    }
    
    // Host output vector
    double *h_c;
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program_mult;               // program
    cl_kernel kernel_mult;                 // kernel
    cl_program program_init;               // program
    cl_kernel kernel_init;                 // kernel
    cl_program program_add;               // program
    cl_kernel kernel_add;                 // kernel
    
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
    // Initialize vectors on host}
 
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 256;
 
    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;
 
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // Create a context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 2,
                            (const char **) & contents.c_str(), NULL, &err);
 
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    //clBuildProgram(program_init, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
     kernel_init = clCreateKernel(program, "matMulKernel", &err);
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel_init, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel_init, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel_init, 1, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel_init, 2, sizeof(unsigned int), &n);

    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(queue, kernel_init, 1, NULL, &globalSize, &localSize,
                                                            0, NULL, NULL);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clEnqueueReadBuffer(queue, d_a, CL_TRUE, 0,
                                bytes, h_a, 0, NULL, NULL );

    clEnqueueReadBuffer(queue, d_b, CL_TRUE, 0,
                                bytes, h_b, 0, NULL, NULL );

    kernel = clCreateKernel(program, "vecAdd", &err);

    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    
    for(int z = 0; z < 1; z++) {
        // Create the input and output arrays in device memory for our calculation
        
        
        // Write our data set into the input array in device memory
        err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                    bytes, h_a, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                    bytes, h_b, 0, NULL, NULL);
    
        // Set the arguments to our compute kernel
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
        err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
    
        // Execute the kernel over the entire range of the data set 
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                                0, NULL, NULL);
    
        // Wait for the command queue to get serviced before reading back results
        clFinish(queue);
    
        // Read the results from the device
        clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                    bytes, h_c, 0, NULL, NULL );
    }
    //Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    
    printf("final result: %f\n", h_c[0]);
 
    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseProgram(program_init);
    clReleaseKernel(kernel_init);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}