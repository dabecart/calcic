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

static void initBlock(BlockHeader *block, size_t blockSize, HeapChunkHeader* currentHeap) 
{
    if(block == NULL) {
        return;
    }

    block->prevFree = block->nextFree = NULL;
    block->size = blockSize;
    block->heap = currentHeap;

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
}

void _insertInFreeList(BlockHeader* block) {
    // This block needs to be inserted along the other free blocks from the same heap.
    if(block->heap->lastFreed == NULL) {
        // No other block from the heap was freed, add this block to the end of the free list.
        block->nextFree = NULL;
        block->prevFree = freeListHead;
        if(freeListHead != NULL) {
            freeListHead->nextFree = block;
        }
        freeListHead = block;
    }else {
        // 'prevInList' will be the previous block to the current one in the free list.
        BlockHeader *prevInList = block->heap->lastFreed;
        BlockHeader *nextInList = prevInList->nextFree;
        while((prevInList != NULL) && (prevInList->heap == block->heap) && (block < prevInList)) {
            nextInList = prevInList;
            prevInList = prevInList->prevFree;
        }

        // Insert the block to the right of the last freed block in the same heap.
        block->prevFree = prevInList;
        block->nextFree = nextInList;

        if(nextInList == NULL) {
            // The 'prevInList' was the head of the free list. Set the head now to be the current 
            // block.
            freeListHead = block;
        }else {
            nextInList->prevFree = block;
        }

        if(prevInList != NULL) {
            prevInList->nextFree = block;
        }
    }
    
    // Store the rightmost freed direction of the heap.
    if(block > block->heap->lastFreed) {
        block->heap->lastFreed = block;
    }

    // printFreeBlocks();
}

void _removeFromFreeList(BlockHeader* block) {
    // If the block being removed is the 'lastFreed' block of the heap, set it to the previous one 
    // in the same heap or NULL if there isn't one.
    if(block == block->heap->lastFreed) {
        if(block->prevFree != NULL && block->prevFree->heap == block->heap) {
            block->heap->lastFreed = block->prevFree;
        }else {
            block->heap->lastFreed = NULL;
        }
    }

    if(block->nextFree == NULL) {
        // This is the head of the free list.
        freeListHead = block->prevFree;
    }else {
        block->nextFree->prevFree = block->prevFree;
    }

    if(block->prevFree != NULL) {
        block->prevFree->nextFree = block->nextFree;
    }

    // printFreeBlocks();
}

void *malloc(size_t size) {
    BlockHeader* block = NULL;

    // Add the size of the block header, and align to BLOCK_ALIGNMENT.
    size = ALIGN_UP(size + BLOCK_HEADER_SIZE, BLOCK_ALIGNMENT);

    if(freeListHead != NULL) {
        // Check if there are any previous blocks which were freed. Start from the free list's head.
        BlockHeader* iterator = freeListHead;
        while(iterator != NULL) {
            if(iterator->size >= size) {
                // We found a previous block which can be reused.
                // Remove 'iterator' from the free list.
                _removeFromFreeList(iterator);

                block = iterator;

                size_t remainingSizeInBlock = iterator->size - size;
                if(remainingSizeInBlock >= (BLOCK_HEADER_SIZE + BLOCK_ALIGNMENT)) {
                    // The block is going to be splitted in two because there's space for a new 
                    // block.
                    BlockHeader *splittedBlock = (BlockHeader*) (((char*) iterator) + size);
                    
                    initBlock(splittedBlock, remainingSizeInBlock, iterator->heap);
                    // Subtract the size of the block to the 'usedSize' in the heap. This is added 
                    // inside initBlock, undo it.
                    block->heap->usedSize -= remainingSizeInBlock;
                    
                    // Insert it into the free list.
                    _insertInFreeList(splittedBlock);

                }else{
                    // If the block is not splitted, make the current block be as large as the 
                    // previous, so no holes are left in the memory.
                    size = iterator->size;
                }

                initBlock(block, size, iterator->heap);

                goto return_malloc;
            }
            
            // Go to the previous block if this one wasn't available.
            iterator = iterator->prevFree;
        }
    }

    // Allocate a new heap block.
    size_t newHeapSize = size + CHUNK_HEADER_SIZE;
    newHeapSize = ALIGN_UP(
        (newHeapSize > HEAP_CHUNK_SIZE) ? newHeapSize : HEAP_CHUNK_SIZE,
        HEAP_CHUNK_SIZE
    );
    HeapChunkHeader *heapChunk = __arch_allocate_memory(newHeapSize);
    if(heapChunk == NULL) {
        return NULL;
    }

    // Modify the total allocated heap size.
    heapSize += newHeapSize;
    initHeap(heapChunk, newHeapSize);

    // The new block will be after the heap chunk.
    block = (BlockHeader*) ((char*) heapChunk + CHUNK_HEADER_SIZE);

    size_t remainingSizeInHeap = heapChunk->size - size;
    if(remainingSizeInHeap >= (BLOCK_HEADER_SIZE + BLOCK_ALIGNMENT)) {
        // We are going to add two blocks to the heap, the requested one and a free one.
        BlockHeader *freeBlock = (BlockHeader*) (((char*) block) + size);
        
        initBlock(freeBlock, remainingSizeInHeap, heapChunk);
        // Subtract the size of the block to the 'usedSize' in the heap. This is added 
        // inside initBlock, undo it.
        heapChunk->usedSize -= remainingSizeInHeap;
        
        // Insert it into the free list.
        _insertInFreeList(freeBlock);

    }else{
        // If there's no space for a free block, make the requested block be as big as the chunk 
        // allows it so there are no holes left in memory.
        size = heapChunk->size;
    }

    // Initialize the block.
    initBlock(block, size, heapChunk);

    if(firstHeapChunk == NULL) {
        // This was the first heap block created. Use this to check if the heap chunk to delete in 
        // free() is the first heap or not. This is used so that we don't keep allocating heap 
        // chunks when only one block is being used by the program.  
        firstHeapChunk = heapChunk;
    }
    
return_malloc:
    return (void*) ((char*) block + BLOCK_HEADER_SIZE);
}
