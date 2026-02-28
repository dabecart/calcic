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
    // Take a look at the previous block.
    if(block->prev != NULL && 
       !block->prev->isInUse &&         // Previous block not in use.
       block->prev->heap == block->heap // Blocks are in the same heap
    ) {
        // Remove the 'prev' block from the free list.
        _removeFromFreeList(block->prev);

        // Join the previous block with the current one. Note that we modify the previous one to 
        // include the current block inside it.
        block->prev->size += block->size;
        block->prev->next = block->next;
        if(block->next != NULL) {
            // If it's null, we're working with the latest block.
            block->next->prev = block->prev;
        }

        if(block == lastBlock) {
            // Update if this block is the last block.
            lastBlock = block->prev;
        }
        block = block->prev;
    }

    if(block->next != NULL && 
       !block->next->isInUse &&         // Next block not in use.
       block->next->heap == block->heap // Blocks are in the same heap
    ) {
        // Remove the 'next' block from the free list.
        _removeFromFreeList(block->next);

        // Join the previous block with the current one. Note that we modify the current one and 
        // include the next block inside the current.
        block->size += block->next->size;
        
        BlockHeader* nextNext = block->next->next;
        block->next = nextNext;
        if(nextNext) {
            nextNext->prev = block;
        }
    }

    // Insert the block into the free list.
    _insertInFreeList(block);

    // Check the last block because it may have become free. If so, go back until you find the last 
    // used block.
    while(lastBlock != NULL && !lastBlock->isInUse) {
        // If the last block is in the free list, we should remove it, as the heap head is going
        // back deallocating every block.
        _removeFromFreeList(lastBlock);

        lastBlock = lastBlock->prev;
    }

    // Check if the heap chunk becomes empty when freeing this block.
    // if(block->heap != firstHeap) {
    //     int isHeapEmpty = 1;

    //     // Check the previous blocks in the same heap as 'block'.
    //     BlockHeader *beforeChunk = block->prev;
    //     while (isHeapEmpty && beforeChunk != NULL && (beforeChunk->heap == block->heap)) {
    //         isHeapEmpty = !beforeChunk->isInUse;
    //         beforeChunk = beforeChunk->prev;
    //     }

    //     // Check the next blocks in the same heap as 'block'.
    //     BlockHeader *afterChunk = block->next;
    //     while (isHeapEmpty && afterChunk != NULL && (afterChunk->heap == block->heap)) {
    //         isHeapEmpty = !afterChunk->isInUse;
    //         afterChunk = afterChunk->next;
    //     }

    //     if(isHeapEmpty) {
    //         if(beforeChunk != NULL && afterChunk != NULL) {
    //             // The chunk being removed is between two chunks, connect them together.
    //             beforeChunk->next = afterChunk;
    //             afterChunk->prev = beforeChunk;

    //         }else if(beforeChunk != NULL) {
    //             // We just removed the latest chunk (lastInChunk is NULL).
    //             beforeChunk->next = NULL;
    //         }

    //         // If the heap chunk is empty, deallocate it.
    //         heapSize -= block->heapSize;
    //         __arch_deallocate_memory(block->heap, block->heapSize);
    //     }
    // }    
}