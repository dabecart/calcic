/***************************************************************************************************
 * strcspn.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

size_t strcspn(const char *s1, const char *s2) {
    const char *sc1, *sc2;

    for(sc1 = s1; *sc1 != 0; sc1++) {
        for(sc2 = s2; *sc2 != 0; sc2++) {
            if(*sc1 == *sc2) {
                // Found the first character that is in both s1 and s2.
                return sc1 - s1;
            }
        }
    }
    // No characters from s1 are in s2.
    return sc1 - s1;
}