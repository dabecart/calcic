/***************************************************************************************************
 * deallocate_memory.c
 * 
 * This function is part of the <arch.h> library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <arch.h>
#include <x64/unistd_64.h>

int __munmap(void *start, size_t len)
{
    int ret;
    
    __asm__("\t"
        "mov        $11, %rax\n\t"      // Move the opcode __NR_munmap to AX.
        "syscall\n\t"                   // AX contains the return value.
        :   "AX" (ret)                  // OUTPUTS
        :   "DI" (start),               // INPUTS (following the order of syscall(2))
            "SI" (len)
    );
    
    return ret;
}

int  __arch_deallocate_memory(void* ptr, size_t size) {
    return __munmap(ptr, size);
}