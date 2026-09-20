/***************************************************************************************************
 * uldiv.c
 * 
 * This function is part of the <stdlib.h> standard library. Optimized for the x64 architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <calcilib.h>

uldiv_t uldiv(unsigned long numer, unsigned long denom) {
    uldiv_t result;

    // The div instruction calculates at the same time the quotient and remainder.
    __asm__("\t"
        "movq     $0, %rdx\n\t"     // Zero out DX.
        "divq     %%r10\n"          // Signed division.
        :   "r:AX"  (result.quot),  // Outputs.
            "r:DX"  (result.rem)
        :   "r:AX"  (numer),        // Inputs.
            "r:R10" (denom)
    );

    return result;
}