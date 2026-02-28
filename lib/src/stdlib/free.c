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

    // The block is now unused.
    block->isInUse = 0;

    // Try to coalesce unused blocks together.
    // We suppose that only one merger must be done and there won't be three unused blocks in a row.
    if(block->prev != NULL && 
       !block->prev->isInUse &&         // Previous block not in use.
       block->prev->heap == block->heap // Blocks are in the same heap
    ) {
        // Join the previous block with the current one. Note that we modify the previous one to 
        // include the current block inside it.
        block->prev->size += block->size;
        block->prev->next = block->next;
        block->next->prev = block->prev;
    }

    else if(block->next != NULL && 
       !block->next->isInUse &&         // Next block not in use.
       block->next->heap == block->heap // Blocks are in the same heap
    ) {
        // Join the previous block with the current one. Note that we modify the current one and 
        // include the next block inside the current.
        block->size += block->next->size;
        block->next->prev = block;
        block->next = block->next->next;
    }

    // Check if the last block is the one we're freeing. If so, go back until you find the last used
    // block.
    if(lastBlock == block) {
        while(lastBlock != NULL && !lastBlock->isInUse && lastBlock != firstHeap) {
            lastBlock = lastBlock->prev;
        }
    }

    // Check if the heap chunk becomes empty when freeing this block.
    if(block->heap != firstHeap) {
        int isHeapEmpty = 1;

        // Check the previous blocks in the same heap as 'block'.
        BlockHeader *beforeChunk = block->prev;
        while (isHeapEmpty && beforeChunk != NULL && (beforeChunk->heap == block->heap)) {
            isHeapEmpty = !beforeChunk->isInUse;
            beforeChunk = beforeChunk->prev;
        }

        // Check the next blocks in the same heap as 'block'.
        BlockHeader *afterChunk = block->next;
        while (isHeapEmpty && afterChunk != NULL && (afterChunk->heap == block->heap)) {
            isHeapEmpty = !afterChunk->isInUse;
            afterChunk = afterChunk->next;
        }

        if(isHeapEmpty) {
            if(beforeChunk != NULL && afterChunk != NULL) {
                // The chunk being removed is between two chunks, connect them together.
                beforeChunk->next = afterChunk;
                afterChunk->prev = beforeChunk;

            }else if(beforeChunk != NULL) {
                // We just removed the latest chunk (lastInChunk is NULL).
                beforeChunk->next = NULL;
            }

            // If the heap chunk is empty, deallocate it.
            __arch_deallocate_memory(ptr, block->size);
        }
    }
    
}