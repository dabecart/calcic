/***************************************************************************************************
 * strspn.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

size_t strspn(const char *s1, const char *s2) {
    const char *sc1, *sc2;

    for(sc1 = s1; *sc1 != 0; sc1++) {
        // If none of the characters from s2 match s1...
        int noneFromS2 = 1;
        for(sc2 = s2; noneFromS2 && (*sc2 != 0); sc2++) {
            noneFromS2 = *sc1 != *sc2;
        }

        if(noneFromS2) {
            break;
        }
    }
    // All characters from s1 are in s2.
    return sc1 - s1;
}