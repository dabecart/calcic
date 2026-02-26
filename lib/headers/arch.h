/***************************************************************************************************
 * <arch.h>
 * 
 * Functions which are dependent on the system architecture. Only used during compilation of the 
 * libraries.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_ARCH_h
#define _CALCIC_ARCH_h

#include <stddef.h>

void *__arch_allocate_memory(size_t size);
int  __arch_deallocate_memory(void* ptr, size_t size);

#endif // _CALCIC_ARCH_h