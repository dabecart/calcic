/***************************************************************************************************
 * calloc.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <stdlib.h>
#include <string.h>

void* calloc(size_t nmemb, size_t size) {
    size_t byteSize = nmemb * size;
    void* ptr = malloc(byteSize);
    
    if(ptr != NULL) {
        // Fill with zeros.
        memset(ptr, 0, byteSize);
    }
    
    return ptr;
}