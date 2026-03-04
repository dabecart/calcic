/***************************************************************************************************
 * isspace.c
 * 
 * This function is part of the <ctype.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_CTYPE
#include <ctype.h>

int isspace(int c) {
    return TYPES(c) & (_CN | _SP);
}
