/***************************************************************************************************
 * strlen.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <string.h>

size_t strlen(const char *s) {
    const char *ps = s;
    
    while(*ps != 0) {
        ps++;
    }
    
    return ps - s;
}