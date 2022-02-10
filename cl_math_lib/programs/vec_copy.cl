__kernel void vcopy( 
   __global const float *a,
   __global float *c,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid];
    }
}