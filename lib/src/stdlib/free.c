/***************************************************************************************************
 * calloc.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>
#include <arch.h>

void free(void *ptr) {
    if(ptr == NULL) {
        return;
    }

    // ptr is expected to be preceded by a block header.
    BlockHeader *block = (BlockHeader*) ((char*) ptr - BLOCK_HEADER_SIZE);

    // Join 'prev' with 'next' as this block is about to dissapear.
    BlockHeader* prev = block->prev;
    BlockHeader* next = block->next;
    if(prev != NULL) {
        prev->next = next;
    }
    if(next != NULL) {
        next->prev = prev;
    }

    // Modify the first and last blocks if necessary.
    if(firstBlock == block) {
        firstBlock = NULL;
    }
    if(lastBlock == block) {
        lastBlock = block->prev;
    }

    // Erase the pointer.
    __arch_deallocate_memory(ptr, block->size);
}