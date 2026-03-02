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
#include <limits.h>

// TODO: div_t, ldiv_t, lldiv_t

#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0

// TODO: MB_CUR_MAX

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Pseudo-random sequence generation.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#define RAND_MAX UINT_MAX
int rand(void);
void srand(unsigned int seed);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Memory management functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

void *calloc(size_t nmemb, size_t size);
void  free(void *ptr);
void *malloc(size_t size);
void *realloc(void *ptr, size_t size);

// All global variables are inside xdecl_stdlib.c
#ifdef COMPILING_STDLIB

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Pseudo-random sequence generation.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
extern unsigned long NEXT_RAND;

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Memory management.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#define ALIGN_UP(n, a) (((n) + ((a) - 1)) & ~((a) - 1))

struct HeapChunkHeader;

// This header is added on the top of the allocated block.
typedef struct BlockHeader{
    // Pointer to the previous and next free block (used for coalescing).
    struct BlockHeader *prevFree, *nextFree;
    // The block's heap chunk position.
    struct HeapChunkHeader* heap;
    // Size of the block (including the header).
    size_t size;
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
extern BlockHeader *freeListHead;
extern size_t heapSize;
extern HeapChunkHeader *chunkDeallocateList[DEALLOCATE_LIST_LEN];
extern int chunkDeallocateLen;

void _insertInFreeList(BlockHeader* block);
void _removeFromFreeList(BlockHeader* block);

#endif // COMPILING_STDLIB

#endif // _CALCIC_STDLIB_h