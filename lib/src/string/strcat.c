/***************************************************************************************************
 * strcat.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

char *strcat(char *s1, char *s2) {
    char *sc1       = (char*) s1;
    const char *sc2 = (const char*) s2;

    // Find the null terminator of sc1.
    while(*sc1 != 0) {
        sc1++;
    }

    // Once found, copy the data from sc2.
    strcpy(sc1, sc2);

    return s1;
}