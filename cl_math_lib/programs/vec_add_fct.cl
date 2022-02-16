__kernel void vadd( 
   __global const float *a,
   __global const float *b,
   __global float *c,
   __global float *d,
   const float l,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] - (l * b[gid]);
        d[gid] = 0.;
    }
}