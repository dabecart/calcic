/***************************************************************************************************
 * atoi.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>

int atoi(const char *nptr) {
    return (int) strtol(nptr, NULL, 10);
}