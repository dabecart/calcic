/***************************************************************************************************
 * <calcilib.h>
 * 
 * Extra functions and macros outside the C99 standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_CALCILIB_h
#define _CALCIC_CALCILIB_h

typedef struct {
    unsigned long quot;
    unsigned long rem;
} uldiv_t;
uldiv_t uldiv(unsigned long numer, unsigned long denom);

typedef struct {
    unsigned int quot;
    unsigned int rem;
} udiv_t;
udiv_t udiv(unsigned int numer, unsigned int denom);

#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))

// Calculates 10^exp fast.
double pow10(int exp);
// Calculates floor(log(x)).
int ilog10(double x);

#endif // _CALCIC_CALCILIB_h