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

// Aligns n to the next multiple of a if n is not a multiple already.
#define ALIGN_UP(n, a) (((n) + ((a) - 1)) & ~((a) - 1))

BlockHeader *firstHeap = NULL;
BlockHeader *lastBlock  = NULL;
BlockHeader *freeListHead = NULL;
size_t heapSize = 0;

static void initHeapBlock(BlockHeader *block, BlockHeader *previousBlock, BlockHeader *nextBlock, 
    size_t blockSize, void* currentHeap, size_t heapSize) 
{
    if(block == NULL) {
        return;
    }

    block->prev = previousBlock;
    block->next = nextBlock;
    block->size = blockSize;
    block->heap = currentHeap;
    block->heapSize = heapSize;
    block->isInUse = 1;

    // Update the previous and next blocks with references to the newly created block.
    if(previousBlock != NULL) {
        previousBlock->next = block;
    }
    if(nextBlock != NULL) {
        nextBlock->prev = block;
    }
}

void _insertInFreeList(BlockHeader* block) {
    block->freeNext = NULL;
    block->freePrev = freeListHead;
    if(freeListHead != NULL) {
        freeListHead->freeNext = block;
    }
    freeListHead = block;
}

void _removeFromFreeList(BlockHeader* block) {
    if(block->freeNext == NULL) {
        // This is the head of the free list.
        freeListHead = block->freePrev;
    }else {
        block->freeNext->freePrev = block->freePrev;
    }

    if(block->freePrev != NULL) {
        block->freePrev->freeNext = block->freeNext;
    }

    block->freeNext = NULL;
    block->freePrev = NULL;
}

void* malloc(size_t size) {
    BlockHeader *block = NULL;
    
    // Add the size of the block header.
    size += BLOCK_HEADER_SIZE;
    
    // Convert 'size' in a multiple of BLOCK_ALIGNMENT.
    size = ALIGN_UP(size, BLOCK_ALIGNMENT);

    // Check if the size fits in the current heap (if it exists).
    if(lastBlock != NULL) {
        char* endOfLastBlock = ((char*) lastBlock) + lastBlock->size;
        char* endOfHeap = ((char*) lastBlock->heap) + lastBlock->heapSize;
        size_t remainingBytesInHeap = endOfHeap - endOfLastBlock;

        if(size <= remainingBytesInHeap) {
            // The new block can be assigned in the current heap next to the last block.
            block = (BlockHeader*) endOfLastBlock;
            initHeapBlock(block, lastBlock, NULL, size, 
                lastBlock->heap, lastBlock->heapSize);

            // This is the last block made in the heap.
            lastBlock = block;
            goto return_malloc;
        }

        // Check if there are any previous blocks which were freed. Start from the free list header.
        BlockHeader* iterator = freeListHead;
        while(iterator != NULL) {
            if(iterator->size >= size) {
                // We found a previous block which can be reused.
                block = iterator;

                // The previous block remains the same.
                BlockHeader *prevBlock = iterator->prev;
                
                // If the size of the current block is the same as the iterator's, then, the 
                // next block also remains the same.
                BlockHeader *nextBlock = iterator->next;
                size_t remainingSizeInBlock = iterator->size - size;
                if(remainingSizeInBlock >= (BLOCK_HEADER_SIZE + BLOCK_ALIGNMENT)) {
                    // The block is going to be splitted in two because there's space for a new 
                    // block.
                    nextBlock = (BlockHeader*) (((char*) iterator) + size);
                    initHeapBlock(nextBlock, block, iterator->next, 
                        remainingSizeInBlock, iterator->heap, iterator->heapSize);
                    
                    // This block is unused.
                    nextBlock->isInUse = 0;

                    // Insert it into the free list.
                    _insertInFreeList(nextBlock);

                }else{
                    // If the block is not splitted, make the current block be as large as the 
                    // previous, so no holes are left in the memory.
                    size = iterator->size;
                }

                // Remove the current block from the free list.
                _removeFromFreeList(block);
                
                initHeapBlock(block, prevBlock, nextBlock, size, iterator->heap, iterator->heapSize);

                goto return_malloc;
            }
            
            // Go to the previous block if this one wasn't available.
            iterator = iterator->freePrev;
        }
    }

    // Allocate a new heap block.
    size_t newHeapSize = ALIGN_UP(
        size > HEAP_CHUNK_SIZE ? size : HEAP_CHUNK_SIZE,
        HEAP_CHUNK_SIZE
    );
    block = __arch_allocate_memory(newHeapSize);
    if(block == NULL) {
        return NULL;
    }

    heapSize += newHeapSize;

    initHeapBlock(block, lastBlock, NULL, size, block, newHeapSize);
    // This is the last block made in the heap.
    lastBlock = block;
    if(firstHeap == NULL) {
        // This was the first heap block created. Use this to check if the heap chunk to delete in 
        // free() is the first heap or not. This is used so that we don't keep allocating heap 
        // chunks when only one block is being used by the program.  
        firstHeap = block;
    }
    
return_malloc:
    return (void*) ((char*) block + BLOCK_HEADER_SIZE);
}