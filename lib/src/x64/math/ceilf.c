/***************************************************************************************************
 * ceilf.c
 * 
 * This function is part of the <math.h> standard library. Optimized for the x64 architecture, using 
 * SSE4.1 instructions.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>

float ceilf(float x) {
    float result;
    
    __asm__ ("\t"
        "roundss   $2, %%xmm0, %%xmm0\n"    // Mode 2: Round towards +inf

        : "r:XMM0" (result)
        : "r:XMM0" (x)
    );

    return result;
}