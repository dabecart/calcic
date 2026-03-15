/***************************************************************************************************
 * strcpy.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

 #include <string.h>

char *strcpy(char* s1, const char *s2) {
    char *sc1       = s1;
    const char *sc2 = s2;

    while(*sc2 != 0){
        *sc1 = *sc2;
        sc1++;
        sc2++;
    };

    // Add the final null termination.
    *sc1 = 0;

    return s1;
}