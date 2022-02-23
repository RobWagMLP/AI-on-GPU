__kernel void conv_3d  (
    __global const float *in, 
    __global const float *kern,
    __global float *out, 
    const int out_x_max, 
    const int out_y_max,
    const int in_x_max , 
    const int in_y_max ,
    const int in_z_max ,
    const int ker_x_max,
    const int ker_y_max)
{
    int x, y, z;

    x = get_global_id(0);
    y = get_global_id(1);
    z = get_global_id(2);

    float temp = 0.0f;
    int inpZ = ( ( z %  in_z_max) * in_x_max * in_y_max );
    int kerZ = z*ker_x_max*ker_y_max;
    for(int posy = 0; posy < ker_y_max; posy++) {
        for(int posx = 0; posx < ker_x_max; posx++) {
            temp += in[ inpZ + ( in_x_max* (y + posy) ) + x + posx ] * kern[ kerZ + ker_x_max*posy + posx ];
        }
    }
    out[ z*out_x_max*out_y_max + out_x_max*y + x ] = temp;
}

__kernel void conv_3d_dw (
    __global const float *in, 
    __global const float *kern,
    __global float *out, 
    const int out_x_max, 
    const int out_y_max,
    const int in_x_max , 
    const int in_y_max ,
    const int in_z_max ,
    const int ker_x_max,
    const int ker_y_max)
{
    int x, y, z;

    x = get_global_id(0);
    y = get_global_id(1);
    z = get_global_id(2);

    float temp = 0.0f;
    int inpZ = ( z % in_z_max ) * in_x_max * in_y_max;
    int kerZ = ( z / in_z_max ) * ker_x_max * ker_y_max;
    for(int posy = 0; posy < ker_y_max; posy++) {
        for(int posx = 0; posx < ker_x_max; posx++) {
            temp += in[ inpZ + ( in_x_max* (y + posy) ) + x + posx ] * kern[ kerZ + ker_x_max*posy + posx ];
        }
    }
    out[ z*out_x_max*out_y_max + out_x_max*y + x ] = temp;
}

__kernel void conv_3d_bwd  (
    __global const float *in, 
    __global const float *kern,
    __global float *out, 
    const int out_x_max, 
    const int out_y_max,
    const int in_x_max , 
    const int in_y_max ,
    const int in_z_max ,
    const int ker_x_max,
    const int ker_y_max)
{
    int x, y, z;

    x = get_global_id(0);
    y = get_global_id(1);
    z = get_global_id(2);

    int inpZ = ( z % in_z_max ) * in_x_max  * in_y_max;
    int kerZ =   z * ker_x_max  * ker_y_max;

    int diffX = out_x_max - in_x_max;
    int diffY = out_y_max - in_y_max;

    int idxYmax = ker_y_max - 1;
    int idxXmax = ker_x_max - 1;

    float temp = 0.0f;
 
    for(int posy = 0; posy < ker_y_max; posy++) {
        int limIdxY = y + posy - diffY;
        if( limIdxY < 0 || limIdxY >= in_y_max ) {
            continue;
        }
        for(int posx = 0; posx < ker_x_max; posx++) {
            int limIdxX = x + posx - diffX;
            if( limIdxX < 0 || limIdxX >= in_x_max ) {
                continue;
            }
            temp += in[ inpZ + ( in_x_max* limIdxY ) + limIdxX ] *kern[ kerZ + ker_x_max*( idxYmax - posy ) + ( idxXmax - posx ) ];
        }
    }
    out[ z * out_x_max * out_y_max + out_x_max * y + x ] = temp;
}