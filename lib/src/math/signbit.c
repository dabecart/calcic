/***************************************************************************************************
 * signbit.c
 * 
 * This function is part of the <math.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>
#include <string.h>

int __builtin_signbit(double x) {
    unsigned long u;
    memcpy(&u, &x, sizeof(u));

    // Return the bit sign.
    return (int)(u >> 63);
}

int __builtin_signbitf(float x) {
    unsigned int u;
    memcpy(&u, &x, sizeof(u));

    // Return the bit sign.
    return (int)(u >> 31);
}