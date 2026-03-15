/***************************************************************************************************
 * <float.h>
 * 
 * Definitions for floating point numbers IEEE 754, according to Section 5.2.4.2.2 of the C99 
 * standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_FLOAT_h
#define _CALCIC_FLOAT_h

// General floating-point characteristics
#define FLT_RADIX       2
#define FLT_ROUNDS      1   // Round to nearest.
#define FLT_EVAL_METHOD 0   // Evaluate to type precision.
#define DECIMAL_DIG     17  // Base-10 digits to represent double without loss.

// Single-precision float (IEEE 754 32-bit)
#define FLT_MANT_DIG    24
#define FLT_DIG         6
#define FLT_MIN_EXP     (-125)
#define FLT_MIN_10_EXP  (-37)
#define FLT_MAX_EXP     128
#define FLT_MAX_10_EXP  38

#define FLT_MAX         3.40282347e+38F
#define FLT_EPSILON     1.19209290e-07F
#define FLT_MIN         1.17549435e-38F

// Double-precision double (IEEE 754 64-bit)
#define DBL_MANT_DIG    53
#define DBL_DIG         15
#define DBL_MIN_EXP     (-1021)
#define DBL_MIN_10_EXP  (-307)
#define DBL_MAX_EXP     1024
#define DBL_MAX_10_EXP  308

#define DBL_MAX         1.7976931348623157e+308
#define DBL_EPSILON     2.2204460492503131e-16
#define DBL_MIN         2.2250738585072014e-308

// TODO: Add long double digits when it is implemented.

#endif // _CALCIC_FLOAT_h