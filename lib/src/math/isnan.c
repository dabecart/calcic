/***************************************************************************************************
 * isnan.c
 * 
 * This function is part of the <math.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>
#include <string.h>

int __builtin_isnan(double x) {
    // A NaN has all ones in the exponent and the mantissa is non-zero.
    unsigned long u;
    memcpy(&u, &x, sizeof(u));
    
    // Exponent mask: 0x7FF0000000000000
    // Mantissa mask: 0x000FFFFFFFFFFFFF
    return ((u & 0x7FF0000000000000UL) == 0x7FF0000000000000UL) && 
           ((u & 0x000FFFFFFFFFFFFFUL) != 0);
}

int __builtin_isnanf(float x) {
    // A NaN has all ones in the exponent and the mantissa is non-zero.
    unsigned int u;
    memcpy(&u, &x, sizeof(u));
    
    // Exponent mask: 0x7FF00000
    // Mantissa mask: 0x000FFFFF
    return ((u & 0x7FF00000U) == 0x7FF00000U) && 
           ((u & 0x000FFFFFU) != 0);
}