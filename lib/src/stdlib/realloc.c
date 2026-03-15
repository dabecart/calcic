/***************************************************************************************************
 * realloc.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <stdlib.h>
#include <string.h>

void* realloc(void *ptr, size_t size) {
    void* new_ptr = malloc(size);

    if(ptr != NULL) {
        // Transfer the data.
        memcpy(new_ptr, ptr, size);
        // Free the pointer.
        free(ptr);
    }

    return new_ptr;
}