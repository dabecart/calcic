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

int __equal_ulong(unsigned long a, unsigned long b);
int __equal_long __attribute__((crude, alias(__equal_ulong))) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "cmp    \n\t"
        "seq    %%r6\n\t"
        "bne    __equal_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "cmp    \n\t"
        "seq    %%r6\n"
    "__equal_long_exit:\n\t"
        "mov    %%r6, %%r0\n\t"
        "ret    \n\t"
    );
}

int __not_equal_ulong(unsigned long a, unsigned long b);
int __not_equal_long __attribute__((crude, alias(__not_equal_ulong))) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "cmp    \n\t"
        "sne    %%r6\n\t"
        "bne    __not_equal_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "cmp    \n\t"
        "sne    %%r6\n"
    "__not_equal_long_exit:\n\t"
        "mov    %%r6, %%r0\n\t"
        "ret    \n\t"
    );
}

int __less_than_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "cmp    \n\t"
        "sls    %%r6\n\t"
        "bne    __less_than_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "cmp    \n\t"
        "sls    %%r6\n"
    "__less_than_long_exit:\n\t"
        "mov    %%r6, %%r0\n\t"
        "ret    \n\t"
    );
}

int __less_or_equal_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "cmp    \n\t"
        "sles   %%r6\n\t"
        "bne    __less_or_equal_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "cmp    \n\t"
        "sles   %%r6\n"
    "__less_or_equal_long_exit:\n\t"
        "mov    %%r6, %%r0\n\t"
        "ret    \n\t"
    );
}

int __greater_than_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "cmp    \n\t"
        "sls    %%r6\n\t"
        "bne    __greater_than_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "cmp    \n\t"
        "sls    %%r6\n"
    "__greater_than_long_exit:\n\t"
        "mov    %%r6, %%r0\n\t"
        "ret    \n\t"
    );
}

int __greater_or_equal_long __attribute__((crude)) (long a, long b) {
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "mov    %%r3, %%op2\n\t"
        "cmp    \n\t"
        "sles   %%r6\n\t"
        "bne    __greater_or_equal_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "mov    %%r2, %%op2\n\t"
        "cmp    \n\t"
        "sles   %%r6\n"
    "__greater_or_equal_long_exit:\n\t"
        "mov    %%r6, %%r0\n\t"
        "ret    \n\t"
    );
}

unsigned int __not_ulong (unsigned long a);
int __not_long __attribute__((crude, alias(__not_ulong))) (long a) {
    // !x is the same as x == 0.
    __asm__("\t"
        "mov    %%r1, %%op1\n\t"
        "clr    %%op2\n\t"
        "cmp    \n\t"
        "seq    %%r6\n\t"
        "bne    __not_long_exit\n\t"
        "mov    %%r0, %%op1\n\t"
        "cmp    \n\t"
        "seq    %%r6\n"
    "__not_long_exit:\n\t"
        "mov    %%r6, %%r0\n\t"
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
        "mov        $1, %%r2\n\t"
        "clr        %%r3\n\t"
        "fun        __sum_long\n\t"
        "ret        \n\t"
    );
}

unsigned long __decrement_ulong(unsigned long a);
long __decrement_long __attribute__((crude, alias(__decrement_ulong))) (long a) {
    // Pass 1 as second argument to __subtract_long.
    __asm__("\t"
        "mov        $1, %%r2\n\t"   
        "clr        %%r3\n\t"
        "fun        __subtract_long\n\t"
        "ret        \n\t"
    );
}


long __multiplication_ulong __attribute__((crude)) (long a, long b) {
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
        "umul       %%r6\n\t"           // bc -> r6
        "mov        %%r1, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "umul       %%op2\n\t"          // ad -> op2
        "mov        %%r6, %%op1\n\t"    // bc -> op1
        "add        %%r4\n\t"           // bc + ad = OP1L -> r4
        "mov        %%r0, %%op1\n\t"
        "mov        %%r2, %%op2\n\t"
        "umul       %%op1, %%r6\n\t"    // bd = OP2H & OP2L -> op1 & r0
        "mov        %%r4, %%op2\n\t"
        "add        %%r1\n\t"           // OP2H + OP1L -> r1
        "ret        \n\t"
    );
}

long __multiplication_long (long a, long b) {
    // Multiply them as unsigned and then calculate the sign. 
    int isANegative = a < 0L;
    int isBNegative = b < 0L;

    if (isANegative) {
        a = -a;
    }
    if (isBNegative) {
        b = -b;
    }
    
    long result = __multiplication_ulong(a, b);

    if (isANegative != isBNegative) {
        result = -result;
    }

    return result;
}

long __signExtend_long __attribute__((crude)) (int a) {
    // If a is negative then write 0xFFFFFFFF to the upper part (r1). The last mov instruction was 
    // used to move the input number, therefore, the negative flag should be set accordingly.
    __asm__("\t"
        "clr    %%r1\n\t"
        "bfnr   __signExtend_long_exit\n\t"
        "mov    $0xFFFFFFFF, %%r1\n\t"
    "__signExtend_long_exit:\n\t"
        "ret    \n\t"
    );
}
