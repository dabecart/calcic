/***************************************************************************************************
 * atol.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>

long atol(const char *nptr) {
    return strtol(nptr, NULL, 10);
}