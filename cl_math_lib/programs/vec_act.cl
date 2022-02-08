__kernel void vact_sig_dw( 
   __global const float *a,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] * (1 - a[gid]);
    }
}

__kernel void vact_sig( 
   __global const float *a,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = 1/(1 + exp(-a[gid]));
    }
}

__kernel void vact_tanh_dw( 
   __global const float *a,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = 1. - (a[gid]*a[gid]);
    }
}

__kernel void vact_tanh( 
   __global const float *a,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = (exp(2.*a[gid] - 1.) )/(exp(2.*a[gid]) + 1.);
    }
}

__kernel void vact_relu_dw( 
   __global const float *a,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        if(a[gid] <= 0.)
            c[gid] = 0.;
        else
            c[gid] = 1.;
    }
}

__kernel void vact_relu( 
   __global const float *a,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        if(a[gid] <= 0.)
            c[gid] = 0.;
        else
            c[gid] = a[gid];
    }
}

__kernel void vact_softmax_dw( 
   __global const float *a,
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
       c[gid] = temp;
    }
}

__kernel void vact_softmax( 
   __global const float *a,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        float temp = 0.;
        for(int i = 0; i < n; i++) {
            temp += exp(a[i]);
        }
        c[gid] = exp(a[gid])/temp;
    }
}