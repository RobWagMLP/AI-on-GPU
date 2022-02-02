__kernel void vadd( 
   __global const float *a,
   __global const float *b,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

__kernel void vmult( 
   __global const float *a,
   __global const float *b,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] * b[gid];
    }
}

__kernel void vsub( 
   __global const float *a,
   __global const float *b,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] - b[gid];
    }
}