/***************************************************************************************************
 * strncpy.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

char *strncpy(char *s1, const char *s2, size_t n) {
    char *sc1       = s1;
    const char *sc2 = s2;

    while((*sc2 != 0) && (n > 0)){
        *sc1 = *sc2;
        sc1++;
        sc2++;
        n--;
    };

    // Add the remaining null terminations.
    while(n > 0) {
        *sc1 = 0;
        sc1++;
        n--;
    }

    return s1;
}