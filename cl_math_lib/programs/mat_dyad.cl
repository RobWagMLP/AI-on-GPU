__kernel void mat_dyad(
    __global const float *a, 
    __global const float *b,
    __global const float *d,
    __global float *c, 
    const int N, 
    const int M)
{
    int i, j, k;
    i = get_global_id(0);
    j = get_global_id(1);
    float temp = 0.0f;
    c[i*M+j] = (a[i] * b[j]) + d[i*M+j];
}
