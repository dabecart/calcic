/***************************************************************************************************
 * div.c
 * 
 * This function is part of the <stdlib.h> standard library. Optimized for the x64 architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>

div_t div(int numer, int denom) {
    div_t result;
    
    // The idiv instruction calculates at the same time the quotient and remainder.
    __asm__("\t"
        "cdq\n\t"                   // Sign extend AX into DX.
        "idivl    %%r10d\n"         // Signed division.
        :   "r:AX"  (result.quot),  // Outputs.
            "r:DX"  (result.rem)
        :   "r:AX"  (numer),        // Inputs.
            "r:R10" (denom)
    );

    return result;
}