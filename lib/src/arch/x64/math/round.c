/***************************************************************************************************
 * round.c
 * 
 * This function is part of the <math.h> standard library. Optimized for the x64 architecture, using 
 * SSE4.1 instructions.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <math.h>

double round(double x) {
    double result;
    
    // 1. Get absolute value of x
    // 2. Add 0.5
    // 3. Truncate (round toward zero)
    // 4. Restore original sign
    __asm__ ("\t"
        "movsd     sign_mask_sd(%%rip), %%xmm1\n\t"
        "andpd     %%xmm0, %%xmm1\n\t"                  // xmm1 = sign bit of x
        
        "movsd     abs_mask_sd(%%rip), %%xmm2\n\t"
        "andpd     %%xmm2, %%xmm0\n\t"                  // xmm0 = |x|
        
        "addsd     half_sd(%%rip), %%xmm0\n\t"          // xmm0 = |x| + 0.5
        
        "roundsd   $3, %%xmm0, %%xmm0\n\t"              // Mode 3: Truncate (round toward zero)
        
        "orpd      %%xmm1, %%xmm0\n"                    // Restore sign

        : "r:XMM0" (result)
        : "r:XMM0" (x)
    );

    return result;
}