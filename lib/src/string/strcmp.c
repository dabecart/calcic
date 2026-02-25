/***************************************************************************************************
 * strcmp.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

int strcmp(const char *s1, const char *s2) {
    // Get the size of the inputs with the null terminator.
    size_t len_s1 = strlen(s1) + 1;
    size_t len_s2 = strlen(s2) + 1;

    // Compare with the minumum of len_s1 and len_s2.
    return memcmp(s1, s2, (len_s1 <= len_s2) ? len_s1 : len_s2);
}
