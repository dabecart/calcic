/***************************************************************************************************
 * long.c
 * 
 * Functions used by the calci32 architecture to manipulate long values.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

unsigned long __sum_ulong(unsigned long a, unsigned long b);
long __sum_long __attribute__((crude, alias(__sum_ulong))) (long a, long b) {
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

unsigned long __subtract_ulong(unsigned long a, unsigned long b);
long __subtract_long __attribute__((crude, alias(__subtract_ulong))) (long a, long b) {
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

int __equal_ulong (unsigned long a, unsigned long b);
int __equal_long __attribute__((crude, alias(__equal_ulong))) (long a, long b) {
    __asm__("\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "seq    %%r0\n\t"
        "bne    __equal_long_exit\n\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "seq    %%r0\n"
    "__equal_long_exit:\n\t"
        "ret    \n\t"
    );
}

int __not_equal_ulong (unsigned long a, unsigned long b);
int __not_equal_long __attribute__((crude, alias(__not_equal_ulong))) (long a, long b) {
    __asm__("\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "sne    %%r0\n\t"
        "bne    __not_equal_long_exit\n\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "sne    %%r0\n"
    "__not_equal_long_exit:\n\t"
        "ret    \n\t"
    );
}

int __less_than_long __attribute__((crude)) (long a, long b) {
    // To compare two signed numbers, the high word is compared using signed operations but the low part is compared
    // using unsigned operations. Example: 0xFFFFFFFF_00000000 < 0xFFFFFFFF_80000000. High words are the same, comparing
    // the low words, we have to do it without sign to make it work.
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "sls    %%r4\n\t"
        "bne    __less_than_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "slu    %%r4\n"
    "__less_than_long_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

int __less_or_equal_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "sles   %%r4\n\t"
        "bne    __less_or_equal_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "sleu   %%r4\n"
    "__less_or_equal_long_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

int __greater_than_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "sgs    %%r4\n\t"
        "bne    __greater_than_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "sgu    %%r4\n"
    "__greater_than_long_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

int __greater_or_equal_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "sges   %%r4\n\t"
        "bne    __greater_or_equal_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "sgeu   %%r4\n"
    "__greater_or_equal_long_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

int __less_than_ulong __attribute__((crude)) (unsigned long a, unsigned long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "slu    %%r4\n\t"
        "bne    __less_than_ulong_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "slu    %%r4\n"
    "__less_than_ulong_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

int __less_or_equal_ulong __attribute__((crude)) (unsigned long a, unsigned long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "sleu   %%r4\n\t"
        "bne    __less_or_equal_ulong_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "sleu   %%r4\n"
    "__less_or_equal_ulong_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

int __greater_than_ulong __attribute__((crude)) (unsigned long a, unsigned long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "sgu    %%r4\n\t"
        "bne    __greater_than_ulong_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "sgu    %%r4\n"
    "__greater_than_ulong_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

int __greater_or_equal_ulong __attribute__((crude)) (unsigned long a, unsigned long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "sub    \n\t"
        "sgeu   %%r4\n\t"
        "bne    __greater_or_equal_ulong_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "sub    \n\t"
        "sgeu   %%r4\n"
    "__greater_or_equal_ulong_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

unsigned int __not_ulong (unsigned long a);
int __not_long __attribute__((crude, alias(__not_ulong))) (long a) {
    // !x is the same as x == 0.
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "clr    %%op2\n\t"
        "sub    \n\t"
        "seq    %%r4\n\t"
        "bne    __not_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "sub    \n\t"
        "seq    %%r4\n"
    "__not_long_exit:\n\t"
        "mov    %%r4, %%r0\n\t"
        "ret    \n\t"
    );
}

unsigned long __bitwise_complement_ulong (unsigned long a);
long __bitwise_complement_long __attribute__((crude, alias(__bitwise_complement_ulong))) (long a) {
    // To negate a number in 2s compliment, invert all bits and add one to the number.
    __asm__("\t"
        "mov        %%r0, %%op1\n\t"
        "not        %%r0\n\t"
        "mov        %%r1, %%op1\n\t"
        "not        %%r1\n\t"
        "ret        \n\t"
    );
}

unsigned long __negation_ulong (unsigned long a);
long __negation_long __attribute__((crude, alias(__negation_ulong))) (long a) {
    // To negate a number in 2s compliment, invert all bits and add one to the number.
    __asm__("\t"
        "fun        __bitwise_complement_long\n\t"
        "fun        __increment_long\n\t"
        "ret        \n\t"
    );
}

unsigned long __increment_ulong(unsigned long a);
long __increment_long __attribute__((crude, alias(__increment_ulong))) (long a) {
    // Pass 1 as second argument to __sum_long.
    __asm__("\t"
        "mov        %%r0, %%op1\n\t"
        "inc        %%r0\n\t"
        "mov        %%r1, %%op1\n\t"
        "clr        %%op2\n\t"
        "addc       %%r1\n\t"
        "ret        \n\t"
    );
}

unsigned long __decrement_ulong(unsigned long a);
long __decrement_long __attribute__((crude, alias(__decrement_ulong))) (long a) {
    // Pass 1 as second argument to __subtract_long.
    __asm__("\t"
        "mov        %%r0, %%op1\n\t"
        "dec        %%r0\n\t"
        "mov        %%r1, %%op1\n\t"
        "clr        %%r3\n\t"
        "subc       %%r1\n\t"
        "ret        \n\t"
    );
}

unsigned long __logic_left_shift_ulong (unsigned long a, int b);
long __logic_left_shift_long __attribute__((crude, alias(__logic_left_shift_ulong))) (long a, int b) {
    // For 0 <= b < 32:     ret_h = (a_h << b) | (a_l >> (32-b)),   ret_l = a_l << b
    // For 32 <= b < 64:    ret_h = a_l << (b - 32),                ret_l = 0
    __asm__("\t"
        "mov        $32, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "sub        %%r4\n\t"                   // 32 - b -> r4
        "bles       __logic_left_over_32\n"
        "mov        %%r1, %%op1\n\t"
        "shl        %%r5\n\t"                   // (a_h << b) -> r5
        "mov        %%r0, %%op1\n\t"
        "shl        %%r6\n\t"                   // (a_l << b) -> r6
        "mov        %%r4, %%op2\n\t"
        "shr        %%op2\n\t"                  // (a_l >> (32-b)) -> op2
        "mov        %%r5, %%op1\n\t"
        "or         %%r1\n\t"
        "mov        %%r6, %%r0\n\t"
        "ret        \n"
    "__logic_left_over_32:\n\t"
        "mov        %%r4, %%op1\n\t"
        "neg        %%op2\n\t"                  // b - 32 -> op2
        "mov        %%r0, %%op1\n\t"
        "shl        %%r1\n\t"                   // a_l << (b - 32) -> r1
        "clr        %%r0\n\t"
        "ret        \n"
    );
}

unsigned long __logic_right_shift_ulong __attribute__((crude)) (unsigned long a, int b) {
    // For 0 <= b < 32:     ret_h = a_h >> b,   ret_l = (a_l >> b) | (a_h << (32-b))
    // For 32 <= b < 64:    ret_h = 0,          ret_l = a_h >> (b - 32)
    __asm__("\t"
        "mov        $32, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "sub        %%r4\n\t"                   // 32 - b -> r4
        "bles       __logic_right_over_32\n"
        "mov        %%r0, %%op1\n\t"
        "shr        %%r5\n\t"                   // (a_l >> b) -> r5
        "mov        %%r1, %%op1\n\t"
        "shr        %%r6\n\t"                   // (a_h >> b) -> r6
        "mov        %%r4, %%op2\n\t"
        "shl        %%op2\n\t"                  // (a_h << (32-b)) -> op2
        "mov        %%r5, %%op1\n\t"
        "or         %%r0\n\t"
        "mov        %%r6, %%r1\n\t"
        "ret        \n"
    "__logic_right_over_32:\n\t"
        "mov        %%r4, %%op1\n\t"
        "neg        %%op2\n\t"                  // b - 32 -> op2
        "mov        %%r1, %%op1\n\t"
        "shr        %%r0\n\t"                   // a_h >> (b - 32) -> r0
        "clr        %%r1\n\t"
        "ret        \n"
    );
}

long __arithmetic_right_shift_long __attribute__((crude)) (long a, int b) {
    // For 0 <= b < 32:     ret_h = a_h shra b,   ret_l = (a_l >> b) | (a_h << (32-b))
    // For 32 <= b < 64:    ret_h = a_h shra 31,  ret_l = a_h shra (b - 32)
    __asm__("\t"
        "mov        $32, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "sub        %%r4\n\t"                   // 32 - b -> r4
        "bles       __arithmetic_right_over_32\n"
        "mov        %%r0, %%op1\n\t"
        "shr        %%r5\n\t"                   // (a_l >> b) -> r5
        "mov        %%r1, %%op1\n\t"
        "shra       %%r6\n\t"                   // (a_h shra b) -> r6
        "mov        %%r4, %%op2\n\t"
        "shl        %%op2\n\t"                  // (a_h << (32-b)) -> op2
        "mov        %%r5, %%op1\n\t"
        "or         %%r0\n\t"
        "mov        %%r6, %%r1\n\t"
        "ret        \n"
    "__arithmetic_right_over_32:\n\t"
        "mov        %%r4, %%op1\n\t"
        "neg        %%op2\n\t"                  // b - 32 -> op2
        "mov        %%r1, %%op1\n\t"
        "shra       %%r0\n\t"                   // a_h shra (b - 32) -> r1
        "mov        $31, %%op2\n\t"
        "shra       %%r1\n\t"                   // a_h shra 31 -> r1
        "ret        \n"
    );
}

unsigned long __multiplication_ulong __attribute__((crude)) (unsigned long a, unsigned long b) {
    /**
        With r0 = b, r1 = a, r2 = d, r3 = c:

        (a*2^32 + b) * (c*2^32 + d) = (ab)*2^64 + (bc + ad)*2^32 + bd
                                       ^OP0        ^OP1            ^OP2

        If we split this sum in 32-bit operations:
                        OP2H    OP2L
                OP1H    OP1L
      + OP0H    OP0L
        -------------------------------
        (out of bounds)|RESH    RESL
     */ 
    __asm__("\t"
        "mov        %%r0, %%op1\n\t"
        "mov        %%r3, %%op2\n\t"
        "umul       %%r4\n\t"           // bc -> r4
        "mov        %%r1, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "umul       %%op2\n\t"          // ad -> op2
        "mov        %%r4, %%op1\n\t"    // bc -> op1
        "add        %%r4\n\t"           // bc + ad = OP1L -> r4
        "mov        %%r0, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "umul       %%r0, %%op1\n\t"    // bd = OP2H & OP2L -> op1 & r0
        "mov        %%r4, %%op2\n\t"
        "add        %%r1\n\t"           // OP2H + OP1L -> r1
        "ret        \n\t"
    );
}

long __multiplication_long (long a, long b) {
    int aNegative = a < 0L;
    int bNegative = b < 0L;

    if (aNegative) a = -a;
    if (bNegative) b = -b;

    unsigned long uresult = __multiplication_ulong((unsigned long) a, (unsigned long) b);

    if (aNegative ^ bNegative) {
        uresult = -uresult;
    }

    return (long) uresult;
}


void __division_algorithm __attribute__((crude)) () {
    /**
     * With A = a*2^32 + b and B = c*2^32 + d, we will calculate A/B and A mod B at once.
     *          ^r1      ^r0       ^r3      ^r2
     */
    __asm__("\t"
        // Check the divisor is not zero.
        "mov        %%r2, %%r4\n\t"
        "bne        __division_algorithm_not_zero_first\n\t"
        "mov        %%r3, %%r5\n\t"
        "bne        __division_algorithm_not_zero_second\n\t"
        "clr        %%op1\n\t"
        "dec        %%r0\n\t"
        "mov        %%r0, %%r1\n\t"
        "mov        %%r0, %%r2\n\t"
        "mov        %%r0, %%r3\n\t"
        "ret        \n"

    "__division_algorithm_not_zero_first:\n\t"
        "mov        %%r3, %%r5\n"

    "__division_algorithm_not_zero_second:\n\t"
    
        "mov        $64, %%r11\n\t"
        "clr        %%r2\n\t"
        "clr        %%r3\n\t"
        
        // Quotient:                [0] = r0, [1] = r1 <- This contains the dividend at the beginning.
        // Remainder:               [0] = r2, [1] = r3
        // Divisor:                 [0] = r4, [1] = r5
    "__division_algorithm_loop_start:\n\t"
        "set        %%op2\n\t"
        "mov        %%r0, %%op1\n\t"
        "shl        %%r0\n\t"
        "mov        %%r1, %%op1\n\t"
        "rol        %%r1\n\t"
        "mov        %%r2, %%op1\n\t"    // Transfer the dividend to the remainder a bit at a time.
        "rol        %%r2\n\t"
        "mov        %%r3, %%op1\n\t"
        "rol        %%r3\n\t"

        "mov        %%r2, %%op1\n\t"    // Try to subtract: remainder - divisor
        "mov        %%r4, %%op2\n\t"
        "sub        %%r6\n\t"
        "mov        %%r3, %%op1\n\t"
        "mov        %%r5, %%op2\n\t"
        "subc       %%r7\n\t"

        "bfcs       __division_algorithm_next_bit\n\t"

        "fun        __increment_long\n\t"   // The subtraction was possible. Increment the quotient.
        "mov        %%r6, %%r2\n\t"         // Move the subtracted values to the remainder.
        "mov        %%r7, %%r3\n"

    "__division_algorithm_next_bit:\n\t"
        "mov        %%r11, %%op1\n\t"
        "dec        %%r11\n\t"
        "bne        __division_algorithm_loop_start\n\t"

        "ret\n\t"
    );
}

unsigned long __division_ulong __attribute__((crude)) (unsigned long a, unsigned long b) {
    __asm__("\t"
        "fun    __division_algorithm\n\t"
        "ret"
    );
}

unsigned long __modulus_ulong __attribute__((crude)) (unsigned long a, unsigned long b) {
    __asm__("\t"
        "fun    __division_algorithm\n\t"
        "mov    %%r2, %%r0\n\t"
        "mov    %%r3, %%r1\n\t"
        "ret    \n\t"
    );
}

long __division_long (long a, long b) {
    int aNegative = a < 0L;
    int bNegative = b < 0L;

    if (aNegative) a = -a;
    if (bNegative) b = -b;

    unsigned long uresult = __division_ulong((unsigned long) a, (unsigned long) b);

    if (aNegative ^ bNegative) {
        uresult = -uresult;
    }

    return (long) uresult;
}

long __modulus_long (long a, long b) {
    int aNegative = a < 0L;

    if (aNegative)  a = -a;
    if (b < 0L)     b = -b;

    unsigned long uresult = __modulus_ulong((unsigned long) a, (unsigned long) b);

    if (aNegative) {
        uresult = -uresult;
    }

    return (long) uresult;
}

long __signExtend_long __attribute__((crude)) (int a) {
    // If a is negative then write 0xFFFFFFFF to the upper part (r1). The last mov instruction was 
    // used to move the input number, therefore, the negative flag should be set accordingly.
    __asm__("\t"
        "clr        %%r1\n\t"
        "bfnr       __signExtend_long_exit\n\t"
        "mov        $0xFFFFFFFF, %%r1\n\t"
    "__signExtend_long_exit:\n\t"
        "ret        \n\t"
    );
}
