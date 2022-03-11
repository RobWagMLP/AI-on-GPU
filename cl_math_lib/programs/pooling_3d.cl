__kernel void pooling_3d  (
    __global const float *in, 
    __global float *outMax,
    __global float *outIdx, 
    const int out_x_max, 
    const int out_y_max,
    const int in_x_max , 
    const int in_y_max ,
    const int pool_dim_x,
    const int pool_dim_y)
{
    int x, y, z;

    x = get_global_id(0);
    y = get_global_id(1);
    z = get_global_id(2);

    int inpZ = ( z  * in_x_max * in_y_max );
    float max = - 1000000;
    int  idxMax = 0;
    for(int posy = y; posy < x + pool_dim_y; posy++) {
        for(int posx = x; posx < x + pool_dim_x; posx++) {
            if( posx < in_x_max && posy < in_y_max) {
                if( in[inpZ + in_x_max*posy + posx] > max) {
                    max     = in[inpZ + in_x_max*posy + posx];
                    idxMax  = inpZ + in_x_max*posy + posx;
                }
            }
        }
    }
    outMax[ z*out_x_max*out_y_max + out_x_max*y + x ] = max;
    outMax[ z*out_x_max*out_y_max + out_x_max*y + x ] = idxMax;
}