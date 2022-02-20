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
    int inpZ = ( z * in_x_max * in_y_max ) % in_z_max;
    int kerZ = z*ker_x_max*ker_y_max;
    for(int posy = 0; posy < ker_y_max; posy++) {
        for(int posx = 0; posx < ker_x_max; posx++) {
            temp += in[ inpZ + ( in_x_max* (y + posy) ) + x + posx ] * kern[ kerZ + ker_x_max*posy + posx ];
        }
    }
    out[ z*out_x_max*out_y_max + out_x_max*y + x ] = temp;
}
