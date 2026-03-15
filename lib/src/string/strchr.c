/***************************************************************************************************
 * strchr.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

char *strchr(const char *s, int c) {
    // The terminating null character is considered to be part of the string.
    size_t len = strlen(s) + 1;

    return memchr(s, c, len);
}