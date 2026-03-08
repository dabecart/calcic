/***************************************************************************************************
 * pow10.c
 * 
 * This function is part of the <calcilib.h> library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <calcilib.h>

// In IEEE 754-1985, the exponent goes from 2^-1022 to 2^1023 which is approximately 10^-307 to 
// 10^307. We'll create two arrays to fast fetch the powers of 10 when parsing the exponents.

// For an exponent z = abs(x):
// 10^z = HIGH_EXP[z / 8] * LOW_EXP[z % 8]
// If x < 0, then calculate 1/10^z.

static double LOW_EXP[8] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7
};
static double HIGH_EXP[39] = {
    1e+000, 1e+008, 1e+016, 1e+024, 1e+032, 1e+040, 1e+048, 1e+056, 
    1e+064, 1e+072, 1e+080, 1e+088, 1e+096, 1e+104, 1e+112, 1e+120, 
    1e+128, 1e+136, 1e+144, 1e+152, 1e+160, 1e+168, 1e+176, 1e+184, 
    1e+192, 1e+200, 1e+208, 1e+216, 1e+224, 1e+232, 1e+240, 1e+248, 
    1e+256, 1e+264, 1e+272, 1e+280, 1e+288, 1e+296, 1e+304, 
};

double pow10(int exponent) {
    double result;
    if(exponent >= 0) {
        result = HIGH_EXP[exponent >> 3] * LOW_EXP[exponent & 0x7];
    }else {
        exponent = -exponent;
        result = HIGH_EXP[exponent >> 3] * LOW_EXP[exponent & 0x7];
        result = 1.0 / result;
    }
    return result;
}
