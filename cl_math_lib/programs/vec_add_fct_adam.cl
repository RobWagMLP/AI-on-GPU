__kernel void vadd_adam( 
   __global float *a,
   __global float *b,
   __global float *w,
   __global float *v,
   const float l,    
   const int n  ,
   const int run)
{
    int gid = get_global_id(0);
    if (gid < n) {
        float wMov =   0.9 * w[gid] + ( 0.1   ) * b[gid];
        float vMov = 0.999 * v[gid] + ( 0.001 ) * b[gid]*b[gid];
        float wMovC = wMov/( 1 - pow(  0.9, run) );
        float vMovC = vMov/( 1 - pow(0.999, run) );
        float temp = a[gid] - wMovC* ( l / ( 0.00001 + sqrt( vMovC) ) );
        a[gid] = temp;
        b[gid] = 0.;
        w[gid] = wMov;
        v[gid] = vMov;
    }
}