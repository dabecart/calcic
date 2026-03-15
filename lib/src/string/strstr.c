/***************************************************************************************************
 * strstr.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

char *strstr(const char *s1, const char *s2) {
    // If s2 points to a string with zero length, the function retrns s1.
    if(*s2 == 0) {
        return (char *) s1;
    }

    for(; *s1 != 0; s1++) {
        const char *sc1 = s1;
        const char *sc2 = s2;

        while((*sc1 == *sc2) && (*sc1 != 0) && (*sc2 != 0)) {
            sc1++;
            sc2++;
        }

        // Reached the end of s2, we got a match.
        if(*sc2 == 0) {
            return (char *) s1;
        }
    }

    // Substring was not found.
    return NULL;
}