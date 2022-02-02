__kernel void smax_cat_crent( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = act[gid] - y[gid];
    }
}

__kernel void sig_cat_crent( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = -y[gid]*(1. - act[gid]);
    }
}

__kernel void tanh_cat_crent( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = -(y[gid]/act[gid]) + (y[gid] *act[gid]);
    }
}

__kernel void relu_cat_crent( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        if(act[gid] > 0. )
            c[gid] = -(y[gid]/act[gid]);
        else
            c[gid] = 0.;
    }
}


__kernel void sig_mult_crent( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = act[gid] - y[gid];
    }
}