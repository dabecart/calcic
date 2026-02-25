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

#include <stddef.h>

// TODO: div_t, ldiv_t, lldiv_t

#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0

// TODO: RAND_MAX, MB_CUR_MAX

void* calloc(size_t nmemb, size_t size);
void  free(void *ptr);
void* malloc(size_t size);
void* realloc(void *ptr, size_t size);

#endif // _CALCIC_STDLIB_h