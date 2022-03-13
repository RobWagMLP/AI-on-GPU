__kernel void pooling_3d_avg  (
    __global const float *in, 
    __global float *outMax,
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
    float sum = -0;
    for(int posy = y; posy < x + pool_dim_y; posy++) {
        for(int posx = x; posx < x + pool_dim_x; posx++) {
            if( posx < in_x_max && posy < in_y_max) {
                sum += in[inpZ + in_x_max*posy + posx];
            }
        }
    }
    outMax[ z*out_x_max*out_y_max + out_x_max*y + x ] = sum/(pool_dim_x * pool_dim_y);
}
