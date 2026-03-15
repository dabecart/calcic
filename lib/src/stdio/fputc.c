/***************************************************************************************************
 * fputc.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>

int fputc(int c, FILE *stream) {
    const unsigned char uc_c = c;
    size_t elems = fwrite(&uc_c, 1, sizeof(uc_c), stream);
    return (elems == sizeof(uc_c)) ? uc_c : EOF;
}
