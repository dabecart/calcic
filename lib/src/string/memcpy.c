/***************************************************************************************************
 * memcpy.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

 #include <string.h>

void *memcpy(void *s1, const void *s2, size_t n) {
    char *sc1       = (char*) s1;
    const char *sc2 = (const char*) s2;

    for(size_t index = 0; index < n; index++) {
        sc1[index] = sc2[index];
    }

    return s1;
}