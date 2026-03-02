/***************************************************************************************************
 * fwrite.c
 * 
 * This function is part of the <stdio.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDIO
#include <stdio.h>
#include <stdlib.h>
#include <arch.h>

size_t fwrite(const void * ptr, size_t size, size_t nmemb, FILE *stream) {
    if((size == 0) || (nmemb == 0) || (stream == NULL)) {
        return 0;
    }

    size_t elementsWritten;

    return elementsWritten;
}
