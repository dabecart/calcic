/***************************************************************************************************
 * udiv.c
 * 
 * This function is part of the <stdlib.h> standard library. Optimized for the x64 architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <calcilib.h>

udiv_t udiv(unsigned int numer, unsigned int denom) {
    udiv_t result;
    
    // The div instruction calculates at the same time the quotient and remainder.
    __asm__("\t"
        "movl     $0, %edx\n\t"     // Zero out DX.
        "divl     %%r10d\n"         // Signed division.
        :   "r:AX"  (result.quot),  // Outputs.
            "r:DX"  (result.rem)
        :   "r:AX"  (numer),        // Inputs.
            "r:R10" (denom)
    );

    return result;
}