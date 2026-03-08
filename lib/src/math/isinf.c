/***************************************************************************************************
 * isinf.c
 * 
 * This function is part of the <math.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>
#include <string.h>

int __builtin_isinf(double x) {
    // An inf has all ones in the exponent and the mantissa is exactly zero.
    unsigned long u;
    memcpy(&u, &x, sizeof(u));
    
    // Do not include the sign.
    return (u & 0x7FFFFFFFFFFFFFFFUL) == 0x7FF0000000000000UL;
}

int __builtin_isinff(float x) {
    // An inf has all ones in the exponent and the mantissa is exactly zero.
    unsigned int u;
    memcpy(&u, &x, sizeof(u));
    
    // Do not include the sign.
    return (u & 0x7F800000U) == 0x7F800000U;
}