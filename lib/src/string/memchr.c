/***************************************************************************************************
 * memchr.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

void *memchr(const void *s, int c, size_t n) {
    const unsigned char *sc = (const unsigned char*) s;
    unsigned char toSearch  = c;

    for(size_t index = 0; index < n; index++) {
        if(*sc == toSearch){
            return (void*) sc;
        }
        // Next character...
        sc++;
    }

    return NULL;
}