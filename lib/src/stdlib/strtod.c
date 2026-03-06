/***************************************************************************************************
 * strtod.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>
#include <errno.h>
#include <float.h>
#include <math.h>

double strtod(const char *nptr, char **endptr) {
    int negative, underflow, overflow;
    double result = _strToDecimal(nptr, endptr, 
        DBL_MIN_10_EXP, DBL_MAX_10_EXP, DBL_MIN, DBL_MAX, &negative, &underflow, &overflow);
    
    if(overflow) {
        // Only overflows trigger ERANGE.
        errno = ERANGE;
        result = HUGE_VAL;
    }else if(negative) {
        result = -result;
    }
    
    return result;
}