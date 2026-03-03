/***************************************************************************************************
 * fputs.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <string.h>

int fputs(const char *s, FILE *stream) {
    size_t len = strlen(s);
    size_t elems = fwrite(s, 1, len, stream);
    return (elems == len) ? len : EOF;
}
