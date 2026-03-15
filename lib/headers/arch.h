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

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Memory allocation (for <stdlib.h>).
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
void    *__arch_allocate_memory(size_t size);
int     __arch_deallocate_memory(void* ptr, size_t size);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// I/O (for <stdio.h>).
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
int     __arch_open_file(const char* path, int flags);
int     __arch_close_file(int fd);
long    __arch_write_file(int fd, const void *buf, size_t count);
long    __arch_read_file(int fd, void *buf, size_t count);

#endif // _CALCIC_ARCH_h