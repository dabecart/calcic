/***************************************************************************************************
 * isgraph.c
 * 
 * This function is part of the <ctype.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_CTYPE
#include <ctype.h>

int isgraph(int c) {
    // True for any printing character (except [space]).
    return TYPES(c) & (_DI | _LO | _UP | _PU | _HX );
}
