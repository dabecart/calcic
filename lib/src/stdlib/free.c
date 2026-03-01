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

    // Subtract the size of the block to the 'usedSize' in the heap.
    block->heap->usedSize -= block->size;

    // Insert the block into the free list.
    _insertInFreeList(block);

    // Try to coalesce unused blocks together. Use the free list to find blocks which are in the 
    // same heap. When a block is coalesced, remove it from the free list and the normal list. Only
    // leave the current block in both lists.

    // Start coalescing to the left.
    BlockHeader *iter = block->prevFree;
    while(iter != NULL &&                               // There's a free block to the left.
        iter->heap == block->heap &&                    // They belong to the same heap.
        ((char*)iter + iter->size) == (char*)block      // They are adjacent.
    ) {   
        // Operate on the previous block and remove the current one.
        block->prevFree->size += block->size;

        // Remove the current block from the free list.
        _removeFromFreeList(block);

        // With this, block->prevFree will be modified for the next iteration.
        block = block->prevFree;
        iter = block->prevFree;
    }

    // Now coalesce to the right.
    iter = block->nextFree;
    while(iter != NULL &&                               // There's a free block to the right.
        iter->heap == block->heap &&                    // They belong to the same heap.
        ((char*)block + block->size) == (char*)iter     // They are adjacent.
    ) {  
        // Operate on the current block and remove the next one.
        block->size += block->nextFree->size;
        
        // Remove the block from the free list.
        _removeFromFreeList(block->nextFree);
    
        iter = block->nextFree;
    }

    // Check the current heap chunk. With the current free() operation, we may have emptied it.
    // Never deallocate the first heap chunk.
    if(block->heap != firstHeapChunk && block->heap->usedSize == 0) {
        // Add this chunk heap to the 'to deallocate' list if it isn't already in it. If the list is
        // full, deallocate the oldest chunk.
        int isInsideFreeList = 0;
        for(int index = 0; index < chunkDeallocateLen; index++) {
            if(chunkDeallocateList[index] == block->heap) {
                isInsideFreeList = 1;
                break;
            }
        }

        if(!isInsideFreeList) {
            if(chunkDeallocateLen == DEALLOCATE_LIST_LEN) {
                // Free the oldest block.
                BlockHeader *iterator = chunkDeallocateList[0]->lastFreed;
                while(iterator != NULL && iterator->heap == chunkDeallocateList[0]) {
                    // Remove it from the free list.
                    _removeFromFreeList(iterator);
        
                    // We made sure that freed blocks are together in the free list, so go back.
                    iterator = iterator->prevFree;
                }
        
                // Modify the total allocated heap size.
                heapSize -= chunkDeallocateList[0]->size + CHUNK_HEADER_SIZE;
                
                // We have removed all blocks in the chunk, deallocate the heap.
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