/***************************************************************************************************
 * close_file.c
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

static int __close(int fd)
{
    int ret;
    
    __asm__("\t"
        "mov        %0, %%rax\n\t"      // Move the opcode __NR_close to AX.
        "syscall    \n\t"               // AX contains the return value.
        :   "r:AX"  (ret)               // OUTPUTS
        :   "i"     (__NR_close),       // INPUTS (following the order of syscall(2))
            "r:DI"  (fd)
    );

    return ret;
}

int __arch_close_file(int fd) {
    return __close(fd);
}