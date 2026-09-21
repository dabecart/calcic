/***************************************************************************************************
 * long.c
 * 
 * Functions used by the calci32 architecture to manipulate long values.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

long __sum_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov        %%r0, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "add        %%r0\n\t"
        "mov        %%r1, %%op1\n\t"
        "mov        %%r3, %%op2\n\t"
        "addc       %%r1\n\t"
        "ret        \n\t"
    );
}

long __subtract_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov        %%r0, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "sub        %%r0\n\t"
        "mov        %%r1, %%op1\n\t"
        "mov        %%r3, %%op2\n\t"
        "subc       %%r1\n\t"
        "ret        \n\t"
    );
}