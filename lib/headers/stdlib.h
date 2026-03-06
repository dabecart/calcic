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

#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0

// TODO: MB_CUR_MAX

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Numeric conversion functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
double atof(const char *nptr);
int atoi(const char *nptr);
long atol(const char *nptr);
double strtod(const char *nptr, char **endptr);
float strtof(const char *nptr, char **endptr);
long strtol(const char *nptr, char **endptr, int base);
unsigned long strtoul(const char *nptr, char **endptr, int base);

// TODO: To be implemented when long double and long long support is added.
// long long atoll(const char *nptr);
// double strtold(const char *nptr, char **endptr);
// long long strtoll(const char *nptr, char **endptr, int base);
// unsigned long long strtoull(const char *nptr, char **endptr, int base);

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

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Integer arithmetic functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
int abs(int j);
long labs(long j);

typedef struct {
    int quot;
    int rem;
} div_t;
div_t div(int numer, int denom);

typedef struct {
    long quot;
    long rem;
} ldiv_t;
ldiv_t ldiv(long numer, long denom);

// TODO: llabs and lldiv to be implemented when long long support is added.

// All global variables are inside xdecl_stdlib.c
#ifdef COMPILING_STDLIB

// Pseudo-random sequence generation.
extern unsigned long NEXT_RAND;

// Memory management.
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

extern HeapChunkHeader *firstHeapChunk;
extern BlockHeader *freeListHead;
extern size_t heapSize;
extern HeapChunkHeader *chunkDeallocateList[DEALLOCATE_LIST_LEN];
extern int chunkDeallocateLen;

void _insertInFreeList(BlockHeader* block);
void _removeFromFreeList(BlockHeader* block);

// Numeric conversion.
#define INFINITY_STR "INFINITY"
#define INF_STR "INF"
#define NAN_STR "NAN"

unsigned long _strToInteger(const char *nptr, char **endptr, int base, int* negative, int *overflow);
double _strToDecimal(const char *nptr, char **endptr,
    const int min10Exp, const int max10Exp, const double minValue, const double maxValue,
    int *negative, int *underflow, int *overflow);

#endif // COMPILING_STDLIB

#endif // _CALCIC_STDLIB_h