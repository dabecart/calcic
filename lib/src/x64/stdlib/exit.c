/***************************************************************************************************
 * exit.c
 * 
 * This function is part of the <stdlib.h> standard library. Optimized for the x64 architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>
#include <x64/unistd_64.h>

void exit(int status) {
    __asm__("\t"
        "mov        %0, %%rax\n\t"      // Move the opcode __NR_exit to AX.
        "syscall    \n\t"               // AX contains the return value.
        :                               // No outputs needed.
        :   "i"     (__NR_exit),        // INPUTS (following the order of syscall(2))
            "r:DI"  (status)
    );
}
