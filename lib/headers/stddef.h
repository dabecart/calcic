/***************************************************************************************************
 * <stddef.h>
 * 
 * General definitions, according to Section 7.17 of the C99 standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_STDDEF_h
#define _CALCIC_STDDEF_h

// Return type of the subtraction of two pointers.
typedef long            ptrdiff_t;
// Return type of the sizeof() function.
typedef unsigned long   size_t;

// wchar not implemented.

// Null pointer.
#define NULL (void*)0

// Returns the offset in bytes to the structure 'member' from the beginning of its structure 'type'.
#define offsetof(type,member) __builtin_offsetof(type,member)

#endif // _CALCIC_STDDEF_h