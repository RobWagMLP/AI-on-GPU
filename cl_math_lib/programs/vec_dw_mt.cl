__kernel void vact_softmax_dw( 
   __global const float *a,
   __global const float *b,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        float temp = 0.;
        float swi = 0;
        for(int i = 0; i < n; i++) {
            if(i == gid) {
                swi = 1;
            } else {
                swi = 0;
            }
            temp += a[gid] * (swi - a[gid]);            
       }
       c[gid] = temp * b[gid];
    }
}

__kernel void vact_relu_dw( 
   __global const float *a,
   __global const float *b,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        if(a[gid] <= 0.)
            c[gid] = 0.;
        else
            c[gid] = b[gid];
    }
}

__kernel void vact_tanh_dw( 
   __global const float *a,
   __global const float *b,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = ( 1. - (a[gid]*a[gid]) )  * b[gid];
    }
}

__kernel void vact_sig_dw( 
   __global const float *a,
   __global const float *b,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = ( a[gid] * (1 - a[gid]) ) * b[gid];
    }
}