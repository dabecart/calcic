/***************************************************************************************************
 * isdigit.c
 * 
 * This function is part of the <ctype.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_CTYPE
#include <ctype.h>

int isdigit(int c) {
    return TYPES(c) & _DI;
}
