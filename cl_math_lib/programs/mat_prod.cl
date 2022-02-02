__kernel void mat_mul(
    __global const float *a, 
    __global const float *b,
    __global float *c, 
    const int N, 
    const int M)
{
    int i, j, k;
    i = get_global_id(0);
    j = get_global_id(1);
    float temp = 0.;
    if(i < N && j < M) {
        for (k = 0; k < N; k++) { 
            temp += a[i*N+k] * b[k*M+j];
        }
        c[i*M+j] = temp;
    }
}
