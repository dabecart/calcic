/***************************************************************************************************
 * roundf.c
 * 
 * This function is part of the <math.h> standard library. Optimized for the x64 architecture, using 
 * SSE4.1 instructions.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>

float roundf(float x) {
    float result;
    
    // 1. Get absolute value of x
    // 2. Add 0.5
    // 3. Truncate (round toward zero)
    // 4. Restore original sign
    __asm__ ("\t"
        "movss     sign_mask_ss(%%rip), %%xmm1\n\t"
        "andps     %%xmm0, %%xmm1\n\t"                  // xmm1 = sign bit of x
        
        "movss     abs_mask_ss(%%rip), %%xmm2\n\t"
        "andps     %%xmm2, %%xmm0\n\t"                  // xmm0 = |x|
        
        "addss     half_ss(%%rip), %%xmm0\n\t"          // xmm0 = |x| + 0.5
        
        "roundss   $3, %%xmm0, %%xmm0\n\t"              // Mode 3: Truncate (round toward zero)
        
        "orps      %%xmm1, %%xmm0\n"                    // Restore sign

        : "r:XMM0" (result)
        : "r:XMM0" (x)
    );

    return result;
}