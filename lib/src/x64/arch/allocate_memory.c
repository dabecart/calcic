/***************************************************************************************************
 * allocate_memory.c
 * 
 * This function is part of the <arch.h> library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <arch.h>
#include <x64/unistd_64.h>

#define PROT_READ     0x1              // Page can be read.      
#define PROT_WRITE    0x2              // Page can be written.      
#define PROT_EXEC     0x4              // Page can be executed.      
#define PROT_NONE     0x0              // Page can not be accessed.      

#define MAP_SHARED            0x01    // Share changes.      
#define MAP_PRIVATE           0x02    // Changes are private.      
#define MAP_SHARED_VALIDATE   0x03    // Share changes and validate extension flags.      
#define MAP_TYPE              0x0f    // Mask for type of mapping.      

#define MAP_FIXED             0x10    // Interpret addr exactly.      
#define MAP_FILE              0
#define MAP_ANONYMOUS         0x20    // Don't use a file.      

#define MAP_FAILED	((void *) -1)

static void *__mmap(void *start, size_t len, int prot, int flags, int fd, long off)
{
    void *ret;
    
    __asm__("\t"
        "mov        %0, %%rax\n\t"      // Move the opcode __NR_mmap to AX.
        "syscall    \n\t"               // AX contains the return value.
        :   "r:AX"  (ret)               // OUTPUTS
        :   "i"     (__NR_mmap),        // INPUTS (following the order of syscall(2))
            "r:DI"  (start),               
            "r:SI"  (len),
            "r:DX"  (prot),
            "r:R10" (flags),
            "r:R8"  (fd),
            "r:R9"  (off)
    );

    return ret;
}

void* __arch_allocate_memory(size_t size) {
    void* ret = __mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    
    if(ret == MAP_FAILED) {
        return NULL;
    }
    return ret;
}