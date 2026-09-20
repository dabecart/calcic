/***************************************************************************************************
 * fabsf.c
 * 
 * This function is part of the <math.h> standard library. Optimized for the x64 architecture, using 
 * SSE4.1 instructions.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>

float fabsf(float x) {
    float result;
    
    __asm__ ("\t"
        "movss     abs_mask_ss(%%rip), %%xmm2\n\t"
        "andps     %%xmm2, %%xmm0\n"                  // xmm0 = |x|
        
        : "r:XMM0" (result)
        : "r:XMM0" (x)
    );

    return result;
}