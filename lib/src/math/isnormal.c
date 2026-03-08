/***************************************************************************************************
 * isnormal.c
 * 
 * This function is part of the <math.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>
#include <string.h>

int __builtin_isnormal(double x) {
    unsigned long u;
    memcpy(&u, &x, sizeof(u));
    
    // Extract the exponent bits (bits 52 to 62).
    int exponent = (u >> 52) & 0x7FF;
    
    // It is normal if: 0 < exponent < 2047.
    return (exponent > 0) && (exponent < 0x7FF);
}

int __builtin_isnormalf(float x) {
    unsigned int u;
    memcpy(&u, &x, sizeof(u));
    
    // Extract the exponent bits (bits 23 to 30).
    int exponent = (u >> 23) & 0xFF;
    
    // It is normal if: 0 < exponent < 256.
    return (exponent > 0) && (exponent < 0xFF);
}