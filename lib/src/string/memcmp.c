/***************************************************************************************************
 * memcmp.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

int memcmp(const void *s1, const void *s2, size_t n) {
    const unsigned char *sc1 = (const unsigned char*) s1;
    const unsigned char *sc2 = (const unsigned char*) s2;

    for(size_t index = 0; index < n; index++) {
        int diff = (int)(sc1[index]) - (int)(sc2[index]);
        if(diff != 0){
            return diff;
        }
    }

    return 0;
}
