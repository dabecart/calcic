/***************************************************************************************************
 * <stdlib.h>
 * 
 * Defines several macros, types and functions of general utility, as defined in Section 7.20 of the
 * C99 standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_STDLIB_h
#define _CALCIC_STDLIB_h

#include <stddef.h> // size_t, NULL

// TODO: div_t, ldiv_t, lldiv_t

#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0

// TODO: RAND_MAX, MB_CUR_MAX

void *calloc(size_t nmemb, size_t size);
void  free(void *ptr);
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);

#ifdef COMPILING_STDLIB

// Aligns n to the next multiple of a if n is not a multiple already.
#define ALIGN_UP(n, a) (((n) + ((a) - 1)) & ~((a) - 1))

// Dynamic memory allocation.

struct HeapChunkHeader;

// This header is added on the top of the allocated block.
// TODO: Optimize this struct.
typedef struct BlockHeader{
    // Pointer to the previous and next block (used in coalescing).
    struct BlockHeader *prev, *next;
    // Pointer to the previous and next free block (used when the block is not in use).
    struct BlockHeader *freePrev, *freeNext;
    // Size of the block (including the header).
    size_t size;

    // The block's heap chunk position.
    struct HeapChunkHeader* heap;

    // Flags.
    char isInUse;
} BlockHeader;

typedef struct HeapChunkHeader {
    // Allocated chunk size (including chunk header).
    size_t size;
    // Used size by the blocks in the chunk. If it reaches zero, this chunk is empty.
    size_t usedSize;
    // Pointer to the last freed block belonging to this chunk.
    BlockHeader *lastFreed;
} HeapChunkHeader;

#define BLOCK_ALIGNMENT     16
#define BLOCK_HEADER_SIZE   ALIGN_UP(sizeof(BlockHeader), BLOCK_ALIGNMENT)
#define CHUNK_HEADER_SIZE   ALIGN_UP(sizeof(HeapChunkHeader), BLOCK_ALIGNMENT)
#define HEAP_CHUNK_SIZE     4096

#define DEALLOCATE_LIST_LEN 5

// Defined in malloc.
extern HeapChunkHeader *firstHeapChunk;
extern BlockHeader *lastBlock;
extern BlockHeader *freeListHead;
extern size_t heapSize;
extern HeapChunkHeader *chunkDeallocateList[DEALLOCATE_LIST_LEN];
extern int chunkDeallocateLen;

void _insertInFreeList(BlockHeader* block);
void _removeFromFreeList(BlockHeader* block);

#endif // COMPILING_STDLIB

#endif // _CALCIC_STDLIB_h