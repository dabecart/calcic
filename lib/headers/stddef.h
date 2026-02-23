/***************************************************************************************************
 * <stddef.h>
 * 
 * General definitions, according to Section 7.17 of the C99 standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
 **************************************************************************************************/

#ifndef _CALCIC_STDDEF_h
#define _CALCIC_STDDEF_h

typedef long            ptrdiff_t;
typedef unsigned long   size_t;

// wchar not implemented.
// typedef int wchar_t;

#define NULL (void*)0

#define offsetof(t,m) __builtin_offsetof(t,m)

#endif // _CALCIC_STDDEF_h