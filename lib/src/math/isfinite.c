/***************************************************************************************************
 * isfinite.c
 * 
 * This function is part of the <math.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>

int __builtin_isfinite(double x) {
    return !__builtin_isinf(x);
}

int __builtin_isfinitef(float x) {
    return !__builtin_isinff(x);
}