/***************************************************************************************************
 * toupper.c
 * 
 * This function is part of the <ctype.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_CTYPE
#include <ctype.h>

int toupper(int c) {
    if(c >= 'a' && c <= 'z') {
        return c - ('a' - 'A');
    }
    
    return c;
}
