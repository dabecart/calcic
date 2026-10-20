/***************************************************************************************************
 * <stdint.h>
 * 
 * Integer types with fixed widths, according to Section 7.18 of the C99 standard. For the x64 
 * architecture.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#ifndef _CALCIC_STDINT_h
#define _CALCIC_STDINT_h

// Exact-width integer types (Section 7.18.1.1)
typedef   signed char  int8_t;
typedef unsigned char  uint8_t;
typedef   signed short int16_t;
typedef unsigned short uint16_t;
typedef   signed int   int32_t;
typedef unsigned int   uint32_t;
typedef   signed long  int64_t;
typedef unsigned long  uint64_t;

#define INT8_MIN    (-128)
#define INT8_MAX    127
#define UINT8_MAX   256

#define INT16_MIN    (-32768)
#define INT16_MAX    32767
#define UINT16_MAX   65535

#define INT32_MIN     (-2147483648)
#define INT32_MAX     2147483647
#define UINT32_MAX    4294967295U

#define INT64_MIN    (-9223372036854775808L)
#define INT64_MAX    9223372036854775807L
#define UINT64_MAX   18446744073709551615UL

// Minimum-width integer types (Section 7.18.1.2)
typedef   signed char  int_least8_t;
typedef unsigned char  uint_least8_t;
typedef   signed short int_least_16_t;
typedef unsigned short uint_least_16_t;
typedef   signed int   int_least_32_t;
typedef unsigned int   uint_least_32_t;
typedef   signed long  int_least_64_t;
typedef unsigned long  uint_least_64_t;

#define INT_LEAST8_MIN    (-128)
#define INT_LEAST8_MAX    127
#define UINT_LEAST8_MAX   256

#define INT_LEAST16_MIN    (-32768)
#define INT_LEAST16_MAX    32767
#define UINT_LEAST16_MAX   65535

#define INT_LEAST32_MIN     (-2147483648)
#define INT_LEAST32_MAX     2147483647
#define UINT_LEAST32_MAX    4294967295U

#define INT_LEAST64_MIN    (-9223372036854775808L)
#define INT_LEAST64_MAX    9223372036854775807L
#define UINT_LEAST64_MAX   18446744073709551615UL

// Sections 7.18.1.3 and 7.18.1.4 are optional.

// Greatest-width integer types (Section 7.18.1.5)
typedef   signed long intmax_t;
typedef unsigned long uintmax_t;

#endif // _CALCIC_STDINT_h