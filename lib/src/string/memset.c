/***************************************************************************************************
 * memset.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

void *memset(void *s, int c, size_t n) {
    unsigned char *sc = (unsigned char *) s;
    unsigned char value = c;

    for(size_t index = 0; index < n; index++) {
        sc[index] = value;
    }
    
    return s;
}