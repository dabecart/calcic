/***************************************************************************************************
 * strncat.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

char *strncat(char *s1, char *s2, size_t n) {
    char *sc1       = (char*) s1;
    const char *sc2 = (const char*) s2;

    // Find the null terminator of sc1.
    while(*sc1 != 0) {
        sc1++;
    }

    // Once found, copy n bytes of data from sc2.
    strncpy(sc1, sc2, n);

    return s1;
}