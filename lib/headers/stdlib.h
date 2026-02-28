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

// Dynamic memory allocation.

// This header is added on the top of the allocated block.
// TODO: Optimize this struct.
typedef struct BlockHeader{
    // Pointer to the previous and next block.
    struct BlockHeader *prev, *next;
    // Size of the block.
    size_t size;

    // The block's heap chunk position.
    struct BlockHeader* heap;
    size_t heapSize;

    // Flags.
    char isInUse;
} BlockHeader;

#define BLOCK_HEADER_SIZE   sizeof(BlockHeader) // bytes
#define BLOCK_ALIGNMENT     16 // bytes
#define HEAP_CHUNK_SIZE     4096 // bytes

// Defined in malloc.
extern BlockHeader *firstHeap;
extern BlockHeader *lastBlock;

#endif // COMPILING_STDLIB

#endif // _CALCIC_STDLIB_h