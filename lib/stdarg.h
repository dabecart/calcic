/***************************************************************************************************
 * <stdarg.h>
 * 
 * This library has the tools required to advance through a list of function arguments whose number
 * and types are not known to the called function when it is translated. Defined in Section 7.15 of
 * the C99 standard. 
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
 **************************************************************************************************/

#ifndef _CALCIC_STDARG_h
#define _CALCIC_STDARG_h

#define va_start(a,b) __builtin_va_start(a,b)
#define va_arg(a,b) __builtin_va_arg(a,b)
#define va_end(a) __builtin_va_end(a)
#define va_copy(a,b) __builtin_va_copy(a,b)

typedef __builtin_va_list va_list;

#endif // _CALCIC_STDARG_h