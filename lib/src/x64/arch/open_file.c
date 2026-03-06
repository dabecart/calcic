/***************************************************************************************************
 * open_file.c
 * 
 * This function is part of the <arch.h> library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <arch.h>
#include <x64/unistd_64.h>
#include <x64/fcntl.h>

#define COMPILING_STDIO
#include <stdio.h>

static int __open(const char *path, int flags)
{
    int ret;
    
    __asm__("\t"
        "mov        %0, %%rax\n\t"      // Move the opcode __NR_open to AX.
        "syscall    \n\t"               // AX contains the return value.
        :   "r:AX"  (ret)               // OUTPUTS
        :   "i"     (__NR_open),        // INPUTS (following the order of syscall(2))
            "r:DI"  (path),
            "r:SI"  (flags)
    );

    return ret;
}

int __arch_open_file(const char* path, int flags) {
    // Convert from the flags in <stdio.h> to <fcntl.h> (POSIX values).
    int openFlags = 0;
    if((flags & READ_MODE) && (flags & WRITE_MODE))     openFlags = O_RDWR;
    else if(flags & READ_MODE)                          openFlags = O_RDONLY;
    else if(flags & WRITE_MODE)                         openFlags = O_WRONLY;

    if(flags & APPEND_MODE) openFlags |= O_APPEND;

    if(flags & TRUNCATE_MODE) openFlags |= O_TRUNC | O_CREAT;

    return __open(path, openFlags);
}