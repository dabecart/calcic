/***************************************************************************************************
 * rand.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>

int rand(void) {
    // Based on the example from Section 7.21.2.2 of the C99 standard.
    NEXT_RAND = NEXT_RAND * 1103515245 + 12345;
    return (unsigned int)(NEXT_RAND >> 16) & 0x7FFF;
}