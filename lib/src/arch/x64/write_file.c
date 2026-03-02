/***************************************************************************************************
 * write_file.c
 * 
 * This function is part of the <arch.h> library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <arch.h>
#include <x64/unistd_64.h>
#include <x64/fcntl.h>

static int __write(int fd, const void *buf, size_t count)
{
    int ret;
    
    __asm__("\t"
        "mov        %0, %%rax\n\t"      // Move the opcode __NR_write to AX.
        "syscall    \n\t"               // AX contains the return value.
        :   "r:AX"  (ret)               // OUTPUTS
        :   "i"     (__NR_write),       // INPUTS (following the order of syscall(2))
            "r:DI"  (fd),
            "r:SI"  (buf),
            "r:DX"  (count)
    );

    return ret;
}

long __arch_write_file(int fd, const void *buf, size_t count) {
    return __write(fd, buf, count);
}