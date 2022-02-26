__kernel void vadd( 
   __global const float *a,
   __global const float *b,
   const float l,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        a[gid] = a[gid] - (l * b[gid]);
        b[gid] = 0.;
    }
}