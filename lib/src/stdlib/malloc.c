/***************************************************************************************************
 * malloc.c
 * 
 * This function is part of the <stdlib.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>
#include <arch.h>

BlockHeader *firstBlock = NULL;
BlockHeader *lastBlock  = NULL;

void* malloc(size_t size) {
    // Add the size of the block header.
    size += BLOCK_HEADER_SIZE;
    
    // Round 'size' with BLOCK_ALIGNMENT.
    size += BLOCK_ALIGNMENT - (size % BLOCK_ALIGNMENT);

    void* p = __arch_allocate_memory(size);
    if(p == NULL) {
        return NULL;
    }

    // Initialize the block header.
    BlockHeader *block = p;
    block->prev = lastBlock;
    block->next = NULL;
    block->size = size;

    // Set the last block's 'next' block to the current one.
    if(lastBlock != NULL) {
        lastBlock->next = block;
    }
    
    lastBlock = block;
    if(firstBlock == NULL) {
        firstBlock = block;
    }
    
    return (void*) ((char*) p + BLOCK_HEADER_SIZE);
}