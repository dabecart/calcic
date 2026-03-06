/***************************************************************************************************
 * atof.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>

double atof(const char *nptr) {
    return strtod(nptr, NULL);
}