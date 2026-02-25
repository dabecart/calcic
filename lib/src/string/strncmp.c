/***************************************************************************************************
 * strncmp.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

int strncmp(const char *s1, const char *s2, size_t n) {
    // Get the size of the inputs with the null terminator.
    size_t len_s1 = strlen(s1) + 1;
    size_t len_s2 = strlen(s2) + 1;

    size_t min_len = (len_s1 <= len_s2) ? len_s1 : len_s2;

    // Compare with the minumum of the shortest input's length and n.
    return memcmp(s1, s2, (min_len <= n) ? min_len : n);
}