__kernel void conv_3d_add  (
    __global const float *in, 
    __global float *out     , 
    const    int   channels ,
    const    int   xmax     ,
    const    int   ymax     )
{
    
    int x, y, z;
    x = get_global_id(0);
    y = get_global_id(1);
    z = get_global_id(2);

    float temp = 0.0f;
    int xy = xmax * y + x;
    for(int i = 0; i < channels; i++) {
        temp += in[i * xmax * ymax + xy];
    }
    out[ z * xmax * ymax + xy ] = temp;
}
