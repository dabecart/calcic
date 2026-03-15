/***************************************************************************************************
 * strtol.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>
#include <limits.h>
#include <errno.h>

long strtol(const char *nptr, char **endptr, int base) {
    int negative, overflow;
    unsigned long num = _strToInteger(nptr, endptr, base, &negative, &overflow);

    long ret = num;
    if(overflow || (num > LONG_MAX)) {
        // Set the errno.
        errno = ERANGE;

        if(negative) {
            ret = LONG_MIN;
        }else {
            ret = LONG_MAX;
        }
    }else if(negative) {
        // Negate the number.
        ret = -ret;
    }

    return ret;
}
