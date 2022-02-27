__kernel void vadd( 
   __global float *a,
   __global float *b,
   const float l,    
   const int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        float temp = a[gid] - (l * b[gid]);
        a[gid] = temp;
        b[gid] = 0.;
    }
}