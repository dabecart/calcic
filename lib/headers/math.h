/***************************************************************************************************
 * <math.h>
 * 
 * Definitions of mathematical funtions and macros, according to Section 7.12 of the C99 standard.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_MATH_h
#define _CALCIC_MATH_h

// Both set to infinity.
#define HUGE_VAL    (1.0)/(0.0)
#define HUGE_VALF   (1.0f)/(0.0f)
// #define HUGE_VALL

#define INFINITY    (1.0f)/(0.0f)
#define NAN         (0.0f)/(0.0f)

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Classification.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
int     __builtin_isfinite(double x);
int     __builtin_isinf(double x);
int     __builtin_isnan(double x);
int     __builtin_isnormal(double x);
int     __builtin_signbit(double x);

int     __builtin_isfinitef(float x);
int     __builtin_isinff(float x);
int     __builtin_isnanf(float x);
int     __builtin_isnormalf(float x);
int     __builtin_signbitf(float x);

#define isfinite(x) \
    (sizeof(x) == sizeof(float)  ? __builtin_isfinitef(x)   : __builtin_isfinite(x))
#define isinf(x) \
    (sizeof(x) == sizeof(float)  ? __builtin_isinff(x)      : __builtin_isinf(x))
#define isnan(x) \
    (sizeof(x) == sizeof(float)  ? __builtin_isnanf(x)      : __builtin_isnan(x))
#define isnormal(x) \
    (sizeof(x) == sizeof(float)  ? __builtin_isnormalf(x)   : __builtin_isnormal(x))
#define signbit(x) \
    (sizeof(x) == sizeof(float)  ? __builtin_signbitf(x)    : __builtin_signbit(x))

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Power and absolute-value functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
double  fabs(double x);
float   fabsf(float x);

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Nearest integer functions.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
double  ceil(double x);
float   ceilf(float x);
double  floor(double x);
float   floorf(float x);
double  round(double x);
float   roundf(float x);
long    lround(double x);
long    lroundf(float x);

#endif // _CALCIC_MATH_h