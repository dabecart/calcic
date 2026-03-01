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

HeapChunkHeader *firstHeapChunk = NULL;
BlockHeader *lastBlock  = NULL;
BlockHeader *freeListHead = NULL;
size_t heapSize = 0;

HeapChunkHeader *chunkDeallocateList[DEALLOCATE_LIST_LEN] = {NULL};
int chunkDeallocateLen = 0;

static void initHeap(HeapChunkHeader *chunk, size_t chunkSize) {
    if(chunk == NULL) {
        return;
    }

    chunk->size = chunkSize - CHUNK_HEADER_SIZE;
    chunk->usedSize = 0;
    chunk->lastFreed = NULL;
}

static void initBlock(BlockHeader *block, BlockHeader *previousBlock, BlockHeader *nextBlock, 
    size_t blockSize, HeapChunkHeader* currentHeap) 
{
    if(block == NULL) {
        return;
    }

    block->prev = previousBlock;
    block->next = nextBlock;

    block->size = blockSize;
    block->heap = currentHeap;
    block->isInUse = 1;

    if(block->heap->usedSize == 0) {
        // The heap chunk is no longer empty. Search for this block in the 'to deallocate' list and 
        // remove it.
        int toDeallocateIndex = 0;
        while(toDeallocateIndex < chunkDeallocateLen) {
            if(block->heap == chunkDeallocateList[toDeallocateIndex]) {
                break;
            }
            toDeallocateIndex++;
        }
        
        if(toDeallocateIndex != chunkDeallocateLen) {
            // If found, move all elements from this index to the left.
            for(int index = toDeallocateIndex; index < chunkDeallocateLen - 1; index++) {
                chunkDeallocateList[index] = chunkDeallocateList[index + 1];
            }
            chunkDeallocateLen--;
        }
    }
    // Add the size of the block to the heap 'usedSize' counter.
    block->heap->usedSize += blockSize;

    // Update the previous and next blocks with references to the newly created block.
    if(previousBlock != NULL) {
        previousBlock->next = block;
    }
    if(nextBlock != NULL) {
        nextBlock->prev = block;
    }
}

void _insertInFreeList(BlockHeader* block) {
    // This block needs to be inserted along the other free blocks from the same heap.
    if(block->heap->lastFreed == NULL) {
        // No other block from the heap was freed, add this block to the end of the free list.
        block->freeNext = NULL;
        block->freePrev = freeListHead;
        if(freeListHead != NULL) {
            freeListHead->freeNext = block;
        }
        freeListHead = block;
    }else {
        // Insert the block to the right of the last freed block in the same heap.
        block->freePrev = block->heap->lastFreed;
        block->freeNext = block->heap->lastFreed->freeNext;

        if(block->heap->lastFreed == freeListHead) {
            // The 'lastFreed' was the head of the free list. Set the head now to be the current 
            // block.
            freeListHead = block;
        }else {
            block->heap->lastFreed->freeNext->freePrev = block;
        }
        block->heap->lastFreed->freeNext = block;
    }
    
    // This is now the last freed block.
    block->heap->lastFreed = block;
    // This block is not in use.
    block->isInUse = 0;
}

void _removeFromFreeList(BlockHeader* block) {
    // If the block being removed is the 'lastFreed' block of the heap, set it to the previous one 
    // in the same heap or NULL if there isn't one.
    if(block == block->heap->lastFreed) {
        if(block->freePrev != NULL && block->freePrev->heap == block->heap) {
            block->heap->lastFreed = block->freePrev;
        }else {
            block->heap->lastFreed = NULL;
        }
    }

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
        char* endOfHeap = ((char*) lastBlock->heap) + lastBlock->heap->size + CHUNK_HEADER_SIZE;
        size_t remainingBytesInHeap = endOfHeap - endOfLastBlock;

        if(size <= remainingBytesInHeap) {
            // The new block can be assigned in the current heap next to the last block.
            block = (BlockHeader*) endOfLastBlock;
            initBlock(block, lastBlock, NULL, size, lastBlock->heap);

            // This is the last block made in the heap.
            lastBlock = block;
            goto return_malloc;
        }

        // Check if there are any previous blocks which were freed. Start from the free list's head.
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
                    initBlock(nextBlock, block, iterator->next, remainingSizeInBlock, iterator->heap);
                    
                    // Insert it into the free list.
                    _insertInFreeList(nextBlock);
                    // Subtract the size of the block to the 'usedSize' in the heap.
                    block->heap->usedSize -= remainingSizeInBlock;

                }else{
                    // If the block is not splitted, make the current block be as large as the 
                    // previous, so no holes are left in the memory.
                    size = iterator->size;
                }

                // Remove the current block from the free list.
                _removeFromFreeList(block);
                
                initBlock(block, prevBlock, nextBlock, size, iterator->heap);

                goto return_malloc;
            }
            
            // Go to the previous block if this one wasn't available.
            iterator = iterator->freePrev;
        }
    }

    // Allocate a new heap block.
    size_t newHeapSize = size + CHUNK_HEADER_SIZE;
    newHeapSize = ALIGN_UP(
        (newHeapSize > HEAP_CHUNK_SIZE) ? newHeapSize : HEAP_CHUNK_SIZE,
        HEAP_CHUNK_SIZE
    );
    HeapChunkHeader *chunk = __arch_allocate_memory(newHeapSize);
    // fprintf(stderr, " [allocated %zu at %p] ", newHeapSize, chunk);
    if(chunk == NULL) {
        return NULL;
    }

    heapSize += newHeapSize;
    initHeap(chunk, newHeapSize);

    // Initialize the block.
    block = (BlockHeader*) ((char*) chunk + CHUNK_HEADER_SIZE);
    initBlock(block, lastBlock, NULL, size, chunk);

    // This is the last block made in the heap.
    lastBlock = block;
    if(firstHeapChunk == NULL) {
        // This was the first heap block created. Use this to check if the heap chunk to delete in 
        // free() is the first heap or not. This is used so that we don't keep allocating heap 
        // chunks when only one block is being used by the program.  
        firstHeapChunk = chunk;
    }
    
return_malloc:
    return (void*) ((char*) block + BLOCK_HEADER_SIZE);
}