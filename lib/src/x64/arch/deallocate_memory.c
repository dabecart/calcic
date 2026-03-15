/***************************************************************************************************
 * deallocate_memory.c
 * 
 * This function is part of the <arch.h> library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <arch.h>
#include <x64/unistd_64.h>

static int __munmap(void *start, size_t len)
{
    int ret;
    
    __asm__("\t"
        "mov        %0, %%rax\n\t"      // Move the opcode __NR_munmap to AX.
        "syscall\n\t"                   // AX contains the return value.
        :   "r:AX"  (ret)               // OUTPUTS
        :   "i"     (__NR_munmap),      // INPUTS (following the order of syscall(2))
            "r:DI"  (start),              
            "r:SI"  (len)
    );
    
    return ret;
}

int  __arch_deallocate_memory(void* ptr, size_t size) {
    return __munmap(ptr, size);
}