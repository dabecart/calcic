/***************************************************************************************************
 * isalpha.c
 * 
 * This function is part of the <ctype.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_CTYPE
#include <ctype.h>

int isalpha(int c) {
    // True if it's a lowercase or uppercase letter.
    return TYPES(c) & (_LO | _UP);
}
