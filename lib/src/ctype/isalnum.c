/***************************************************************************************************
 * isalnum.c
 * 
 * This function is part of the <ctype.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_CTYPE
#include <ctype.h>

int isalnum(int c) {
    // True if it's a digit, lowercase or uppercase letter.
    return TYPES(c) & (_DI | _LO | _UP);
}
