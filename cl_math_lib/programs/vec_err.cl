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

__kernel void sig_mean_squared( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = act[gid]*(1 - act[gid])*(act[gid] - y[gid]);
    }
}


__kernel void tanh_mean_squared( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = (1 - ( act[gid] * act[gid] ) ) * ( act[gid] - y[gid] );
    }
}

__kernel void relu_mean_squared( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        if(act[gid] > 0. ) {
            c[gid] = ( act[gid] - y[gid] );
        }
        else {
            c[gid] = 0.;
        }
    }
}


__kernel void smax_mean_squared( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        float temp = 0.;
        float swi = 0.;
        for(int i = 0; i < n; i++) {
            if(i == gid) {
                swi = 1.;
            } else {
                swi = 0.;
            }
            temp += ( act[gid] - y[gid] ) * act[gid] * ( swi - act[gid] );           
        }
       c[gid] = temp;
    }
}

__kernel void tanh_mult_crent( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] =  ( ( act[gid] - y[gid] ) / act[gid] ) * ( 1 + act[gid] );
    }
}


__kernel void relu_mult_crent( 
   __global const float *act,
   __global const float *y,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        if( act[gid] > 0. ) {
            c[gid] =  ( act[gid] - y[gid] ) / ( act[gid] * ( 1 - act[gid] ) );
        } else {
            c[gid] = 0.;
        }
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