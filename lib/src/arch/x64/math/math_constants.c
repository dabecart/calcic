/***************************************************************************************************
 * math_constants.c
 * 
 * Constants used in the <math.h> functions optimized for the x64 architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

static void math_constants() {
    __asm__ ("\t"
        ".section .rodata\n\t"
        ".align 16\n\t"

        // To extract the sign of a double.
        ".globl sign_mask_sd\n\t"
        "sign_mask_sd: .quad 0x8000000000000000\n\t"
        // To mask the sign bit of a double.
        ".globl abs_mask_sd\n\t"
        "abs_mask_sd:  .quad 0x7FFFFFFFFFFFFFFF\n\t"
        // (double) 0.5
        ".globl half_sd\n\t"
        "half_sd:      .double 0.5\n\t"

        // To extract the sign of a double.
        ".globl sign_mask_ss\n\t"
        "sign_mask_ss: .quad 0x80000000\n\t"
        // To mask the sign bit of a double.
        ".globl abs_mask_ss\n\t"
        "abs_mask_ss:  .quad 0x7FFFFFFF\n\t"
        // (float) 0.5
        ".globl half_ss\n\t"
        "half_ss:      .float 0.5\n\t"

        ".section .text\n"
    );
}