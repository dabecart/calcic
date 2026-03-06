/***************************************************************************************************
 * strtoul.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>
#include <errno.h>

unsigned long strtoul(const char *nptr, char **endptr, int base) {
    int negative, overflow;
    unsigned long num = _strToInteger(nptr, endptr, base, &negative, &overflow);

    if(overflow) {
        // Set the errno.
        errno = ERANGE;
        num = ULONG_MAX;
    }else if(negative) {
        // Negate the number.
        num = -num;
    }

    return num;
}
