/***************************************************************************************************
 * fgetc.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>

int fgetc(FILE *stream) {
    unsigned char c;
    size_t elems = fread(&c, 1, sizeof(c), stream);
    return (elems == sizeof(c)) ? c : EOF;
}
