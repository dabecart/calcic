/***************************************************************************************************
 * ldiv.c
 * 
 * This function is part of the <stdlib.h> standard library. Optimized for the x64 architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>

ldiv_t ldiv(long numer, long denom) {
    ldiv_t result;

    // The idiv instruction calculates at the same time the quotient and remainder.
    __asm__("\t"
        "cqo\n\t"                   // Sign extend AX into DX.
        "idivq    %%r10\n"          // Signed division.
        :   "r:AX"  (result.quot),  // Outputs.
            "r:DX"  (result.rem)
        :   "r:AX"  (numer),        // Inputs.
            "r:R10" (denom)
    );

    return result;
}