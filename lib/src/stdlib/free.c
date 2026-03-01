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
    // fprintf(stderr, "free(%p)\n", block);

    // Subtract the size of the block to the 'usedSize' in the heap.
    block->heap->usedSize -= block->size;

    // Try to coalesce unused blocks together.
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

    if(lastBlock != NULL) {
        // In case we removed a bunch of blocks when pruning free blocks, remember to set the last 
        // block next to NULL.
        lastBlock->next = NULL;
    }
    
    // Check the current heap chunk. With the current free() operation, we may have emptied it.
    // Never deallocate the first heap chunk.
    if(block->heap != firstHeapChunk && block->heap->usedSize == 0) {
        // Add this chunk heap to the 'to deallocate' list if it isn't already in it. If the list is
        // full, deallocate the oldest chunk.
        int isInsideList = 0;
        for(int index = 0; index < chunkDeallocateLen; index++) {
            if(chunkDeallocateList[index] == block->heap) {
                isInsideList = 1;
                break;
            }
        }

        if(!isInsideList) {
            if(chunkDeallocateLen == DEALLOCATE_LIST_LEN) {
                // Free the oldest block.
                BlockHeader *iterator = chunkDeallocateList[0]->lastFreed;
                while(iterator != NULL && iterator->heap == chunkDeallocateList[0]) {
                    // Remove it from the free list.
                    _removeFromFreeList(iterator);
        
                    // Remove it from the normal list.
                    if(iterator->next != NULL) {
                        iterator->next->prev = iterator->prev;
                    }
                    if(iterator->prev != NULL) {
                        iterator->prev->next = iterator->next;
                    }
        
                    // We made sure that freed blocks are together in the list, so go back in the 
                    // list.
                    iterator = iterator->freePrev;
                }
        
                // We have removed all blocks in the chunk, deallocate the heap.
                heapSize -= chunkDeallocateList[0]->size + CHUNK_HEADER_SIZE;
                // fprintf(stderr, "deallocated %p\n", chunkDeallocateList[0]);
                __arch_deallocate_memory(chunkDeallocateList[0], chunkDeallocateList[0]->size + CHUNK_HEADER_SIZE);
                
                for(int index = 0; index < DEALLOCATE_LIST_LEN - 1; index++) {
                    chunkDeallocateList[index] = chunkDeallocateList[index + 1];
                }
                chunkDeallocateLen--;
            }
    
            // Add this chunk to the list.
            chunkDeallocateList[chunkDeallocateLen] = block->heap;
            chunkDeallocateLen++;
        }
    }
}